# Extracted from vLLM 0.19.1 InputBatch.swap_states; Apache-2.0.
def swap_states(self, i1: int, i2: int) -> None:
    old_id_i1 = self._req_ids[i1]
    old_id_i2 = self._req_ids[i2]
    i1_active_token_count = self._get_active_token_count(i1)
    i2_active_token_count = self._get_active_token_count(i2)
    max_active_token_count = max(i1_active_token_count, i2_active_token_count)
    (self._req_ids[i1], self._req_ids[i2]) = (self._req_ids[i2], self._req_ids[i1])
    (self.req_output_token_ids[i1], self.req_output_token_ids[i2]) = (self.req_output_token_ids[i2], self.req_output_token_ids[i1])
    (self.spec_token_ids[i1], self.spec_token_ids[i2]) = (self.spec_token_ids[i2], self.spec_token_ids[i1])
    assert old_id_i1 is not None and old_id_i2 is not None
    (self.req_id_to_index[old_id_i1], self.req_id_to_index[old_id_i2]) = (self.req_id_to_index[old_id_i2], self.req_id_to_index[old_id_i1])
    (self.num_tokens_no_spec[i1], self.num_tokens_no_spec[i2]) = (self.num_tokens_no_spec[i2], self.num_tokens_no_spec[i1])
    (self.num_prompt_tokens[i1], self.num_prompt_tokens[i2]) = (self.num_prompt_tokens[i2], self.num_prompt_tokens[i1])
    (self.num_computed_tokens_cpu[i1], self.num_computed_tokens_cpu[i2]) = (self.num_computed_tokens_cpu[i2], self.num_computed_tokens_cpu[i1])
    tmp_token_ids = self.token_ids_cpu[i1, :max_active_token_count].copy()
    self.token_ids_cpu[i1, :max_active_token_count] = self.token_ids_cpu[i2, :max_active_token_count]
    self.token_ids_cpu[i2, :max_active_token_count] = tmp_token_ids
    self.is_token_ids[[i1, i2], :max_active_token_count] = self.is_token_ids[[i2, i1], :max_active_token_count]
    embeds_i1 = self.req_prompt_embeds.get(i1)
    embeds_i2 = self.req_prompt_embeds.get(i2)
    if embeds_i1 is not None:
        self.req_prompt_embeds[i2] = embeds_i1
    else:
        self.req_prompt_embeds.pop(i2, None)
    if embeds_i2 is not None:
        self.req_prompt_embeds[i1] = embeds_i2
    else:
        self.req_prompt_embeds.pop(i1, None)
    self.block_table.swap_row(i1, i2)
    (self.request_lora_mapping[i1], self.request_lora_mapping[i2]) = (self.request_lora_mapping[i2], self.request_lora_mapping[i1])
    if self.is_pooling_model:
        return
    self.batch_update_builder.moved.append((i1, i2, MoveDirectionality.SWAP))
    (self.temperature_cpu[i1], self.temperature_cpu[i2]) = (self.temperature_cpu[i2], self.temperature_cpu[i1])
    (self.top_p_cpu[i1], self.top_p_cpu[i2]) = (self.top_p_cpu[i2], self.top_p_cpu[i1])
    (self.top_k_cpu[i1], self.top_k_cpu[i2]) = (self.top_k_cpu[i2], self.top_k_cpu[i1])
    (self.frequency_penalties_cpu[i1], self.frequency_penalties_cpu[i2]) = (self.frequency_penalties_cpu[i2], self.frequency_penalties_cpu[i1])
    (self.presence_penalties_cpu[i1], self.presence_penalties_cpu[i2]) = (self.presence_penalties_cpu[i2], self.presence_penalties_cpu[i1])
    (self.repetition_penalties_cpu[i1], self.repetition_penalties_cpu[i2]) = (self.repetition_penalties_cpu[i2], self.repetition_penalties_cpu[i1])
    (self.num_accepted_tokens_cpu[i1], self.num_accepted_tokens_cpu[i2]) = (self.num_accepted_tokens_cpu[i2], self.num_accepted_tokens_cpu[i1])
    swap_dict_values(self.generators, i1, i2)
    swap_dict_values(self.bad_words_token_ids, i1, i2)
    if self.allowed_token_ids_mask_cpu_tensor is not None:
        (self.allowed_token_ids_mask_cpu_tensor[i1], self.allowed_token_ids_mask_cpu_tensor[i2]) = (self.allowed_token_ids_mask_cpu_tensor[i2], self.allowed_token_ids_mask_cpu_tensor[i1])
