mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");

    // 加载 Llama 模型
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);

    // 初始化分词器
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    // 文本输入
    let input = "Once upon a time";

    // 将文本转化为 token-id 序列
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();

    // 文本生成
    let output_ids = llama.generate(
        input_ids,
        500,
        0.9,
        4,
        1.,
    );

    // 解码
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}
