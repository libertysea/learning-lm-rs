mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;
use kvcache::KVCache;
use std::io::Write;
use std::io::{self, BufRead, BufReader};
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn main() {
    println!();
    println!("请开始输入（输入'exit'退出）：");
    println!();
    println!("<|im_start|> system: \n This is a story continuation model \n <|im_end|> ");

    // 加载模型
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    let cache: Option<kvcache::KVCache<f32>> = None;
    let mut value: (Vec<u32>, kvcache::KVCache<f32>);
    let mut output_ids: Vec<u32>;
    let mut new_cache: KVCache<f32> = llama.new_cache();
    let mut tmp_ids: Vec<u32> = Vec::new();

    // 使用BufReader包装stdin
    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin);
    
    let im_start = "<|im_start|>";  
    let im_end = "<|im_end|>";  
    let mut output:String;
  
    

    // 无限循环，直到用户输入'exit'

    loop {
        // 先打印前缀
        println!();
        println!("<|im_start|> system-prompt: ");
        // 注意这里使用std::io::stdout().flush()来确保前缀立即被打印出来
        io::stdout().flush().expect("无法刷新stdout");

        // 读取一行输入
        let mut input = String::new();
        if reader.read_line(&mut input).is_err() {
            eprintln!("读取输入时出错");
            break;
        }

        println!(" <|im_end|> \n");

        // 检查是否输入了'exit'
        if input.trim() == "exit" {
            println!("退出程序。");
            break;
        }

        // 拼接字符串以形成最终的message  
        let message = format!("{} system-prompt \n {}{}", im_start, input, im_end);  
        // println!("{}", message);

        let binding = tokenizer.encode(message, true).unwrap();
        let input_ids = binding.get_ids();

        tmp_ids.extend_from_slice(&input_ids);

        if cache.is_none() {
            value = llama.generate_chat(&tmp_ids, 500, 0.9, 4, 1., None);
            output_ids = value.0;
            new_cache = value.1;
        } else {
            value = llama.generate_chat(&tmp_ids, 500, 0.9, 4, 1., Some(new_cache));
            output_ids = value.0;
            new_cache = value.1;
        }

        tmp_ids.extend_from_slice(&output_ids);


        output =  tokenizer.decode(&output_ids, true).unwrap().trim_start().to_string();
        let first_space_or_newline = output.find(|c: char| c.is_whitespace() || c == '\n').unwrap_or(output.len());  

        let role = &output[..first_space_or_newline].trim_end().to_string(); 
        let out = &output[first_space_or_newline..].trim_start().to_string();

        println!("{} {} \n {}\n{}", im_start,role, out, im_end)
    
    }
}
