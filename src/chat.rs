use std::sync::Arc;

use crate::{kvcache::KVCache, model::Llama};

pub struct Chat<'a> {
    id: usize,
    message: String,
    his_message: String,
    input_ids: Vec<u32>,
    re_input_ids: Vec<u32>,
    model: &'a Llama<f32>,
    cache: Arc<KVCache<f32>>,
    h_cache: Arc<KVCache<f32>>,
}

impl<'a> Chat<'a> {
    pub fn new_chat(id: usize, model: &'a Llama<f32>) -> Chat<'a> {
        let message = String::new();
        let his_message = String::new();
        let input_ids = Vec::new();
        let re_input_ids = Vec::new();
        let cache = Arc::new(model.new_cache());
        let h_cache = Arc::new(model.new_cache());

        Chat {
            id,
            message,
            his_message,
            input_ids,
            re_input_ids,
            model: &model,
            cache,
            h_cache,
        }
    }

    pub fn generate(
        model: &'a Llama<f32>,
        message: String,
        input: &[u32],
        cache: Arc<KVCache<f32>>,
    ) -> (Vec<u32>, KVCache<f32>) {
        let value;
        let kv_cache = Arc::into_inner(cache);

        if message.is_empty() {
            value = model.generate_chat(&input, 100, 0.9, 4, 1., None);
        } else {
            value = model.generate_chat(&input, 100, 0.9, 4, 1., kv_cache);
        }
        value
    }

    pub fn get_message(&self) -> &String {
        &self.message
    }

    pub fn get_ids(&self) -> Vec<u32> {
        self.input_ids.clone()
    }

    
    pub fn get_re_ids(&self) -> Vec<u32> {
        self.re_input_ids.clone()
    }

    pub fn get_his_message(&self) -> &String {
        &self.his_message
    }

    pub fn get_cache(&self) -> Arc<KVCache<f32>> {
        self.cache.clone()
    }

    pub fn get_h_cache(&self) -> Arc<KVCache<f32>> {
        self.h_cache.clone()
    }

    pub fn set_his_cache(&mut self, his_cache: Arc<KVCache<f32>>) {
        self.h_cache = his_cache;
    }

    pub fn set_cache(&mut self, cache: Arc<KVCache<f32>>) {
        self.cache = cache;
    }

    pub fn set_message(&mut self, message: String) {
        self.message = message;
    }

    pub fn set_input_ids(&mut self, input_ids: Vec<u32>) {
        self.input_ids = input_ids;
    }

    pub fn set_re_input_ids(&mut self, re_input_ids: Vec<u32>) {
        self.re_input_ids = re_input_ids
    }

    pub fn set_his_message(&mut self, his_message: String) {
        self.his_message = his_message;
    }
}