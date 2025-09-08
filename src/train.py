import math
import argparse
import os
import sys
import tqdm
import inspect
import logging
import random
import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from model import ImplicitModel
from configuration_model import ImplicitModelConfig
from data import CoTDataset, CoTDataCollator, extract_answer
from utils import get_sep_position, batch_ids, save_model
from torch.nn.utils.rnn import pad_sequence


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING)


def get_punctuation_token_ids(tokenizer):
    """Get token IDs for common punctuation marks."""
    punctuation_chars = ['.', ',', ';', ':', '!', '?', '\n']
    punctuation_tokens = set()
    
    for punct in punctuation_chars:
        # Handle different tokenization patterns
        tokens = tokenizer.encode(punct, add_special_tokens=False)
        punctuation_tokens.update(tokens)
        
        # Also check with spaces (some tokenizers treat " ." differently than ".")
        tokens_with_space = tokenizer.encode(f' {punct}', add_special_tokens=False)
        punctuation_tokens.update(tokens_with_space)
    
    return punctuation_tokens


def chunk_text(text, punctuation_chars=['.', ',', ';', ':', '!', '?']):
    """Chunks text using punctuation."""
    chunks = []
    current_start = 0
    
    for i, char in enumerate(text):
        if char in punctuation_chars:
            # Include the punctuation in the chunk
            chunk_text = text[current_start:i+1]
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append(chunk_text)
            current_start = i + 1
    
    # Handle remaining text (if no punctuation at end)
    if current_start < len(text):
        remaining = text[current_start:]
        if remaining.strip():
            chunks.append(remaining)
    
    return chunks


def remove_chunks(input_ids, labels, tokenizer, first_sep_pos, second_sep_pos, 
                                    eos_position, chunks_to_remove, removal_side, 
                                    punctuation_chars=['.', ',', ';', ':', '!', '?']):
    """Remove chunks from text."""
    
    # Step 1: Extract reasoning section tokens
    reasoning_tokens = input_ids[first_sep_pos + 1:second_sep_pos]
    
    if len(reasoning_tokens) == 0:
        return input_ids[:eos_position+1], labels[:eos_position+1]
    
    # Step 2: Decode to text
    reasoning_text = tokenizer.decode(reasoning_tokens, skip_special_tokens=True)
    
    # Step 3: Chunk the text
    chunks = chunk_text(reasoning_text, punctuation_chars)
    
    if len(chunks) == 0 or chunks_to_remove == 0:
        return input_ids[:eos_position+1], labels[:eos_position+1]
    
    # Step 4: Determine which chunks to keep
    if removal_side == 'left':
        chunks_to_remove_actual = min(chunks_to_remove, len(chunks))
        if chunks_to_remove_actual > 0:
            # Keep chunks after the removed ones
            chunks_to_keep = chunks[chunks_to_remove_actual:]
        else:
            chunks_to_keep = chunks
    else:  # 'right'
        chunks_to_remove_actual = min(chunks_to_remove, len(chunks))
        if chunks_to_remove_actual > 0:
            # Keep chunks before the removed ones
            chunks_to_keep = chunks[:-chunks_to_remove_actual]
        else:
            chunks_to_keep = chunks
    
    # Step 5: Reconstruct the reasoning text
    if chunks_to_keep:
        # Join chunks with appropriate spacing
        reconstructed_chunks = []
        for i, chunk in enumerate(chunks_to_keep):
            if i > 0:
                # Add space between chunks if the previous chunk doesn't end with space
                # and current chunk doesn't start with space
                prev_chunk = reconstructed_chunks[-1] if reconstructed_chunks else ""
                if prev_chunk and not prev_chunk.endswith(' ') and not chunk.startswith(' '):
                    reconstructed_chunks.append(' ')
            reconstructed_chunks.append(chunk)
        new_reasoning_text = ''.join(reconstructed_chunks)
    else:
        new_reasoning_text = ""
    
    # Step 6: Re-tokenize the new reasoning
    if new_reasoning_text.strip():
        new_reasoning_tokens = tokenizer.encode(new_reasoning_text, add_special_tokens=False)
        new_reasoning_tensor = torch.tensor(new_reasoning_tokens, device=input_ids.device, dtype=input_ids.dtype)
    else:
        new_reasoning_tensor = torch.tensor([], device=input_ids.device, dtype=input_ids.dtype)
    
    # Step 7: Reconstruct full sequence
    new_input_ids = torch.cat([
        input_ids[:first_sep_pos + 1],  # Question + first separator
        new_reasoning_tensor,           # Modified reasoning
        input_ids[second_sep_pos:eos_position+1]  # Second separator + answer + end
    ])
    
    # Handle labels similarly: make reasoning labels align 1-to-1 with new_reasoning_tensor
    # original reasoning labels slice (may be longer than new reasoning tokens)
    orig_reasoning_labels = labels[first_sep_pos + 1:second_sep_pos]

    # number of tokens in new reasoning (0 if empty)
    new_reasoning_len = new_reasoning_tensor.size(0) if new_reasoning_tensor.numel() > 0 else 0

    if new_reasoning_len == 0:
        reasoning_labels = torch.tensor([], device=labels.device, dtype=labels.dtype)
    else:
        # If original reasoning labels are longer, truncate them.
        # If original reasoning labels are shorter (shouldn't normally happen), pad with -100.
        if orig_reasoning_labels.size(0) >= new_reasoning_len:
            reasoning_labels = orig_reasoning_labels[:new_reasoning_len]
        else:
            # pad with -100 for ignored tokens to reach required length
            pad_len = new_reasoning_len - orig_reasoning_labels.size(0)
            pad_tensor = torch.full((pad_len,), -100, device=labels.device, dtype=labels.dtype)
            reasoning_labels = torch.cat([orig_reasoning_labels, pad_tensor], dim=0)

    new_labels = torch.cat([
        labels[:first_sep_pos + 1],
        reasoning_labels,
        labels[second_sep_pos:eos_position+1]
    ])

    
    return new_input_ids, new_labels


def find_punctuation_positions(input_ids, punctuation_token_ids, start_pos, end_pos):
    """Find positions of punctuation marks within a range."""
    positions = []
    for i in range(start_pos, min(end_pos, len(input_ids))):
        if input_ids[i].item() in punctuation_token_ids:
            positions.append(i)
    return positions


def compute_chunks_to_remove_distribution(removal_smoothing_lambda, max_chunks=20):
    """Compute distribution for number of chunks to remove."""
    if removal_smoothing_lambda == float('inf'):
        chunks_distribution = torch.zeros(max_chunks)
        chunks_distribution[0] = 1
    else:
        positions = torch.arange(max_chunks)
        chunks_distribution = (1 - math.exp(-removal_smoothing_lambda)) * positions.mul(-removal_smoothing_lambda).exp()
        cum_prob = chunks_distribution.sum()
        assert cum_prob <= 1
        chunks_distribution[-1] = chunks_distribution[-1] + (1-cum_prob)
    return chunks_distribution

    
@torch.no_grad()
def evaluate(dataloader, tokenizer, device, ctx, model, max_new_tokens, scheduled_chunks_to_remove, removal_side, removal_smoothing_lambda, chunks_distribution, punctuation_token_ids, keep_position=False, disable_random_removal_offset=False):
    model.eval()
    total_instances = 0
    total_tokens = 0
    total_correct = 0
    total_correct_tokens = 0
    total_loss = 0
    position_ids_all = None
    position_ids = None
    
    for batch in tqdm.tqdm(dataloader):
        input_ids_all = batch['input_ids_all'].to(device)
        labels = batch['labels_all'].to(device)
        # Remove answer part
        sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
        input_ids = input_ids_all[:, :sep_positions.max()+1]
        batch_size = input_ids.shape[0]
        first_sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
        second_sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id, skip=1)
        eos_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id, skip=2)

        if scheduled_chunks_to_remove > 0 or removal_smoothing_lambda != float('inf'):
            input_ids_all_tmp = []
            labels_tmp = []
            random_removal_offset = torch.multinomial(chunks_distribution, batch_size, replacement=True).to(device)
            if disable_random_removal_offset:
                random_removal_offset.fill_(0)
            chunks_to_remove = scheduled_chunks_to_remove + random_removal_offset

            for batch_id in range(input_ids_all.shape[0]):
                eos_position = eos_positions[batch_id]
                first_sep_pos = first_sep_positions[batch_id]
                second_sep_pos = second_sep_positions[batch_id]
                chunks_to_remove_this = chunks_to_remove[batch_id].item()
                
                # Use text-level chunking
                new_input_ids, new_labels = remove_chunks(
                    input_ids_all[batch_id], labels[batch_id], tokenizer,
                    first_sep_pos, second_sep_pos, eos_position,
                    chunks_to_remove_this, removal_side
                )
                
                input_ids_all_tmp.append(new_input_ids)
                labels_tmp.append(new_labels)
            
            input_ids_all = pad_sequence(input_ids_all_tmp, batch_first=True, padding_value=tokenizer.eos_token_id).to(device)
            labels = pad_sequence(labels_tmp, batch_first=True, padding_value=-100).to(device)

        with ctx:
            if keep_position:
                position_ids_all = torch.arange(0, input_ids_all.shape[-1], dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
                position_ids_all = position_ids_all[:, :input_ids_all.shape[-1]]
            outputs = model.compute_loss(input_ids=input_ids_all, labels=labels, position_ids=position_ids_all)

        total_loss += outputs.total_loss.item()
        total_correct_tokens += outputs.total_correct.item()
        total_tokens += outputs.total_tokens
        total_instances += batch_size

        # Generate
        stop_on_two_eos = True
        if keep_position:
            position_ids = position_ids_all[:, :input_ids.shape[-1]]
        beam_output = model.generate(
            input_ids=input_ids,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            stop_on_two_eos=stop_on_two_eos,
        )

        # Evaluate
        for i, (input_ids_all_i, beam_output_i) in enumerate(zip(input_ids_all, beam_output)):
            sep_position = sep_positions[i].item()
            tgt = input_ids_all_i[sep_position+1:]
            tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
            ans = extract_answer(tgt_text)
            pred_text = tokenizer.decode(beam_output_i[0][sep_position+1:], skip_special_tokens=True)
            pred_ans = extract_answer(pred_text)
            if ans == pred_ans:
                total_correct += 1
            print (f'Input: {tokenizer.decode(input_ids_all_i[:sep_position], skip_special_tokens=True)}')
            print (f'Target: {tgt_text}')
            print (f'Predicted: {pred_text}')
            print ('')
    
    accuracy = total_correct / total_instances
    token_accuracy = total_correct_tokens / total_tokens
    loss = total_loss / total_tokens
    ppl = math.exp(loss)
    return accuracy, token_accuracy, ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--remove_per_epoch', type=float, default=2)  # Now chunks per epoch, not tokens
    parser.add_argument('--remove_all_when_remove_beyond', type=str, default='inf')
    parser.add_argument('--removal_smoothing_lambda', type=float, default=float('inf'))
    parser.add_argument('--removal_side', type=str, choices=['left', 'right'], default='left')
    parser.add_argument('--pretrain_epochs', type=int, default=0)
    parser.add_argument('--truncation', type=int, default=-1)
    parser.add_argument('--max_len_train', type=int, default=-1)
    parser.add_argument('--max_new_tokens', type=int, default=800)
    parser.add_argument('--max_size', type=int, default=-1)
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--from_pretrained', type=str, default=None)
    parser.add_argument('--remove_start_from', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--bf16', action='store_true')
    parser.set_defaults(bf16=False)
    parser.add_argument('--reset_optimizer', action='store_true')
    parser.set_defaults(reset_optimizer=False)
    parser.add_argument('--keep_position', action='store_true')
    parser.set_defaults(keep_position=False)
    parser.add_argument('--reinitialize_weights', action='store_true')
    parser.set_defaults(reinitialize_weights=False)
    args = parser.parse_args()

    if args.remove_all_when_remove_beyond == 'inf':
        args.remove_all_when_remove_beyond = float('inf')
    else:
        args.remove_all_when_remove_beyond = int(args.remove_all_when_remove_beyond)
    
    print(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    chunks_distribution = compute_chunks_to_remove_distribution(args.removal_smoothing_lambda)
    print("Chunks distribution:", chunks_distribution.tolist()[:10])

    dtype = 'float32'
    if args.bf16:
        dtype = 'bfloat16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    print(ptdtype, dtype, device)

    # Create model
    if args.from_pretrained is None:
        config = ImplicitModelConfig(base_model=args.model)
        model = ImplicitModel(config).to(device).to(ptdtype)
    else:
        print(f'Loading from {args.from_pretrained}')
        model = ImplicitModel.from_pretrained(args.from_pretrained).to(device).to(ptdtype)
    
    if 'gpt2' in args.model:
        old_length = model.base_model.transformer.wpe.weight.shape[0]
        if args.truncation > old_length and args.from_pretrained is None:
            print('EXPANDING POSITIONs')
            new_wpe = torch.nn.Embedding(args.truncation, model.base_model.transformer.wpe.weight.shape[-1])
            new_wpe.weight.data[:old_length] = model.base_model.transformer.wpe.weight
            new_wpe.weight.data[old_length:] = model.base_model.transformer.wpe.weight[-1].view(1, -1).expand(args.truncation-old_length, -1)
            model.base_model.transformer.wpe = new_wpe

            for block in model.base_model.transformer.h:
                block.attn.register_buffer(
                    "bias",
                    torch.tril(torch.ones((args.truncation, args.truncation), dtype=torch.bool)).view(
                        1, 1, args.truncation, args.truncation
                ),
                persistent=False,
            )
    
    model = model.to(device).to(ptdtype)
    tokenizer = model.tokenizer

    # Get punctuation token IDs
    punctuation_token_ids = get_punctuation_token_ids(tokenizer)
    print(f"Punctuation token IDs: {punctuation_token_ids}")

    if args.reinitialize_weights:
        print('reinitializing weights')
        model.base_model.apply(model.base_model._init_weights)

    if args.keep_position:
        assert 'gpt2' in args.model  # only implemented for gpt2 generate

    # Load data
    collate_fn = CoTDataCollator(tokenizer)
    train_dataset = CoTDataset(tokenizer, args.train_path, args.truncation, max_size=args.max_size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataset = CoTDataset(tokenizer, args.val_path, args.truncation)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    if args.test_path:
        test_dataset = CoTDataset(tokenizer, args.test_path, args.truncation)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    # Create Optimizer
    trainable_params = list(model.parameters())
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)

    # Train
    step = 0
    scheduled_chunks_to_remove = 0
    if args.remove_start_from > 0:
        print(f'the number of removed CoT chunks starts from {args.remove_start_from}')
        scheduled_chunks_to_remove = args.remove_start_from

    position_ids = None

    steps_per_epoch = len(train_dataloader)
    steps_per_removed_chunk = int(round(steps_per_epoch / args.remove_per_epoch))
    remove_step_counter = 0
    best_val_accuracy = float('-inf')

    all_cot_removed_in_prev_batch = False
    for epoch in range(args.epochs):
        if scheduled_chunks_to_remove < float('inf'):
            scheduled_chunks_to_remove = int(round(scheduled_chunks_to_remove))
        if scheduled_chunks_to_remove >= args.remove_all_when_remove_beyond:
            scheduled_chunks_to_remove = float('inf')  # remove all
        print(f"Epoch {epoch}. Scheduled chunks to remove: {scheduled_chunks_to_remove}")
        model.train()

        batch_counter = 0
        for batch in tqdm.tqdm(train_dataloader):
            batch_counter += 1
            prev_scheduled_chunks_to_remove = scheduled_chunks_to_remove
            if remove_step_counter == steps_per_removed_chunk or steps_per_removed_chunk == 0:
                scheduled_chunks_to_remove += 1
                remove_step_counter = 0
            if epoch >= args.pretrain_epochs:
                remove_step_counter += 1
            if scheduled_chunks_to_remove > prev_scheduled_chunks_to_remove:
                print(f" -epoch {epoch}. step {step}. removing chunks: {scheduled_chunks_to_remove}")
                if args.reset_optimizer and (not all_cot_removed_in_prev_batch):
                    print('RESETTING OPTIMIZER')
                    optimizer.zero_grad(set_to_none=True)
                    del optimizer
                    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)
            if scheduled_chunks_to_remove >= args.remove_all_when_remove_beyond:
                scheduled_chunks_to_remove = float('inf')  # remove all
            
            input_ids = batch['input_ids_all'].to(device)
            labels = batch['labels_all'].to(device)
            batch_size = input_ids.shape[0]

            first_sep_positions = get_sep_position(input_ids, tokenizer.eos_token_id)
            second_sep_positions = get_sep_position(input_ids, tokenizer.eos_token_id, skip=1)
            eos_positions = get_sep_position(input_ids, tokenizer.eos_token_id, skip=2)

            all_cot_removed_in_batch = False
            if scheduled_chunks_to_remove > 0 or args.removal_smoothing_lambda != float('inf'):
                input_ids_tmp = []
                labels_tmp = []
                random_removal_offset = torch.multinomial(chunks_distribution, batch_size, replacement=True).to(device)
                chunks_to_remove = scheduled_chunks_to_remove + random_removal_offset
                if epoch < args.pretrain_epochs:
                    chunks_to_remove.fill_(args.remove_start_from)
                
                if args.keep_position:
                    position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)

                all_cot_removed_in_batch = True
                for batch_id in range(input_ids.shape[0]):
                    eos_position = eos_positions[batch_id]
                    first_sep_pos = first_sep_positions[batch_id]
                    second_sep_pos = second_sep_positions[batch_id]
                    chunks_to_remove_this = chunks_to_remove[batch_id].item()
                    
                    # Debug output for first 3 examples of first batch
                    if batch_counter <= 5 and batch_id < 3:
                        # Show reasoning section before removal
                        reasoning_tokens = input_ids[batch_id][first_sep_pos + 1:second_sep_pos]
                        reasoning_text = tokenizer.decode(reasoning_tokens, skip_special_tokens=True)
                        
                        # Show chunks found and what will be removed
                        chunks = chunk_text(reasoning_text, ['.', ',', ';', ':', '!', '?'])
                        
                        if len(chunks) > 0 and chunks_to_remove_this > 0:
                            if args.removal_side == 'left':
                                chunks_to_remove_actual = min(chunks_to_remove_this, len(chunks))
                                removed_chunks = chunks[:chunks_to_remove_actual] if chunks_to_remove_actual > 0 else []
                            else:
                                chunks_to_remove_actual = min(chunks_to_remove_this, len(chunks))
                                removed_chunks = chunks[-chunks_to_remove_actual:] if chunks_to_remove_actual > 0 else []
                        else:
                            removed_chunks = []
                        
                        print(f"\n--- Batch {batch_counter}, Example {batch_id + 1} ---")
                        print(f"CURRENT: {reasoning_text}")
                        if removed_chunks:
                            print(f"REMOVING: {''.join(removed_chunks)}")
                        else:
                            print(f"REMOVING: (nothing)")
                    
                    # Use text-level chunking
                    new_input_ids, new_labels = remove_chunks(
                        input_ids[batch_id], labels[batch_id], tokenizer,
                        first_sep_pos, second_sep_pos, eos_position,
                        chunks_to_remove_this, args.removal_side
                    )
                    
                    # Continue debug output for first 3 examples of first batch
                    if batch_counter <= 5 and batch_id < 3:
                        new_first_sep = None
                        new_second_sep = None
                        sep_count = 0
                        for i, token_id in enumerate(new_input_ids):
                            if token_id == tokenizer.eos_token_id:
                                if sep_count == 0:
                                    new_first_sep = i
                                elif sep_count == 1:
                                    new_second_sep = i
                                    break
                                sep_count += 1
                        
                        if new_first_sep is not None and new_second_sep is not None:
                            new_reasoning_tokens = new_input_ids[new_first_sep + 1:new_second_sep]
                            new_reasoning_text = tokenizer.decode(new_reasoning_tokens, skip_special_tokens=True)
                            print(f"REMAINING: {new_reasoning_text}")
                        else:
                            print("REMAINING: (none)")
                        print()

                    # Check if we still have reasoning content after removal
                    reasoning_start_new = first_sep_pos + 1
                    reasoning_end_new = None
                    # Find new second separator position
                    for i in range(reasoning_start_new, len(new_input_ids)):
                        if new_input_ids[i] == tokenizer.eos_token_id:
                            reasoning_end_new = i
                            break
                    
                    if reasoning_end_new is not None and reasoning_end_new > reasoning_start_new:
                        all_cot_removed_in_batch = False
                    
                    input_ids_tmp.append(new_input_ids)
                    labels_tmp.append(new_labels)
                    
                    if args.keep_position:
                        # Adjust position IDs for the shortened sequence
                        original_length = input_ids.shape[-1]
                        new_length = len(new_input_ids)
                        length_diff = original_length - new_length
                        if length_diff > 0:
                            position_ids[batch_id, first_sep_pos:] = torch.arange(
                                first_sep_pos, first_sep_pos + (original_length - first_sep_pos) - length_diff,
                                dtype=torch.long, device=device
                            )
                
                input_ids = batch_ids(input_ids_tmp, tokenizer.eos_token_id, device, input_ids.dtype)
                labels = batch_ids(labels_tmp, -100, device, input_ids.dtype)
                if not all_cot_removed_in_batch:
                    best_val_accuracy = float('-inf')
            
            all_cot_removed_in_prev_batch = all_cot_removed_in_batch
            if args.max_len_train > 0 and input_ids.shape[-1] > args.max_len_train:
                print('skipped')
                continue
        
            with ctx:
                if args.keep_position:
                    position_ids = position_ids[:, :input_ids.shape[-1]]
                outputs = model.compute_loss(input_ids=input_ids, labels=labels, position_ids=position_ids)
            loss = outputs.loss
            loss.div(args.accumulate).backward()
            if step % args.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 100 == 0:
                token_accuracy = outputs.token_accuracy.item()
                ppl = loss.exp().item()
                print(f"Step: {step}. PPL: {ppl}. Token Accuracy: {token_accuracy}")
            step += 1
        
        print(f'Scheduled chunks to remove: {scheduled_chunks_to_remove}')
        accuracy, token_accuracy, ppl = evaluate(
            val_dataloader, tokenizer, device, ctx, model, args.max_new_tokens, 
            scheduled_chunks_to_remove, args.removal_side, args.removal_smoothing_lambda, 
            chunks_distribution, punctuation_token_ids, keep_position=args.keep_position, 
            disable_random_removal_offset=True
        )
        print(f'Disable Offset Val. PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}.')
        
        if accuracy > best_val_accuracy:
            print('***best so far or removed more CoT chunks***')
            best_val_accuracy = accuracy
            if args.test_path:
                accuracy, token_accuracy, ppl = evaluate(
                    test_dataloader, tokenizer, device, ctx, model, args.max_new_tokens,
                    scheduled_chunks_to_remove, args.removal_side, args.removal_smoothing_lambda,
                    chunks_distribution, punctuation_token_ids, keep_position=args.keep_position,
                    disable_random_removal_offset=True
                )
                print(f'Test. PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}.')
        
        EPOCHS_PER_CHECKPOINT = 32
        if (epoch + 1) % EPOCHS_PER_CHECKPOINT == 0:   
            model.save_pretrained(os.path.join(args.save_model, f'checkpoint_{epoch+1}'))

if __name__ == "__main__":
    main()
