mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;
use chat::Chat;
use kvcache::KVCache;
use std::io::Write;
use std::io::{self, BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;
mod chat;

use std::str::FromStr;

fn main() {
    println!();
    println!("请开始输入 \n 输入 exit 退出 \n 输入 new chat 创建新的对话 \n 输入 change chat-id 切换对话 \n 输入 regenerate 重新生成文本\n 输入 history 查看历史对话");
    println!();
    println!("<|im_start|> system: \n This is a story continuation model \n <|im_end|> ");

    // 加载模型
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    let mut output_ids: Vec<u32>;
    let mut tmp_ids: Vec<u32> = Vec::new();
    let mut re_input_ids: Vec<u32> = Vec::new();
    let mut tmp_cache: Arc<KVCache<f32>>;

    // 使用BufReader包装stdin
    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin);

    let im_start = "<|im_start|>";
    let im_end = "<|im_end|>";

    let mut chat_vec: Vec<Chat> = Vec::new();

    let mut id: usize = 0;
    let mut change_id: usize;

    let mut output: String;
    let mut mss: String;
    let mut message_input: String;
    let mut message: String;
    let mut message_new: String;
    let mut his_message_new: String;

    // 无限循环，直到用户输入'exit'

    // 默认创建 chat_0
    let chat_0 = Chat::new_chat(id, &llama);
    chat_vec.push(chat_0);
    change_id = 0;

    loop {
        // 先打印前缀
        println!();
        println!("Please input: ");
        // 注意这里使用std::io::stdout().flush()来确保前缀立即被打印出来
        io::stdout().flush().expect("无法刷新stdout");

        // 读取一行输入
        let mut input = String::new();
        if reader.read_line(&mut input).is_err() {
            eprintln!("读取输入时出错");
            break;
        }

        println!();

        // 检查是否输入 'exit'
        if input.trim() == "exit" {
            println!("退出程序。");
            break;
        }


        // 检查是否输入 'new chat'
        if input.trim() == "new chat" {
            id += 1;
            let new_chat = Chat::new_chat(id, &llama);
            chat_vec.push(new_chat);
            println!("create a new chat, chat-{}", id);
            change_id = id;
            continue;
        }

        // 检查是否输入 'change chat-id'
        if input.starts_with("change") {
            // 尝试获取编号（跳过 "change" 和可能的空格）
            let id_part = input.split_whitespace().skip(1).next();
            if let Ok(c_id) = usize::from_str(id_part.unwrap()) {
                change_id = c_id;
                println!("change chat-{}", change_id);
            } else {
                println!("The input is invalid, please enter again.")
            }
            continue;
        }


        let chat_tmp = &mut chat_vec[change_id];

        // 检查是否输入 'change chat-id'
        if input.trim() == "history" {
            let history = chat_tmp.get_message();
            println!("{}", history);
            continue;
        }

        // 检查是否输入 'regenerate'
        if input.trim() == "regenerate" {
            mss = chat_tmp.get_message().to_string();
            tmp_cache = chat_tmp.get_h_cache();
            tmp_ids = chat_tmp.get_re_ids();
            re_input_ids = chat_tmp.get_re_ids();
            message_input = chat_tmp.get_his_message().to_string();
            message = chat_tmp.get_his_message().to_string();
        } else {
            mss = chat_tmp.get_message().to_string();
            tmp_cache = chat_tmp.get_cache();
            tmp_ids = chat_tmp.get_ids();
            re_input_ids = chat_tmp.get_ids();
            message_input = format!("{} system-prompt \n {}{}", im_start, input, im_end);
            message = chat_tmp.get_message().to_string();
        }

        let binding = tokenizer.encode(message_input.clone(), true).unwrap();
        let input_ids = binding.get_ids();

        tmp_ids.extend_from_slice(&input_ids);
        re_input_ids.extend_from_slice(&input_ids);

        let value = Chat::generate(&llama, mss.to_string(), &tmp_ids, tmp_cache.clone());
        let new_cache = Arc::new(value.1);

        output_ids = value.0;
        tmp_ids.extend_from_slice(&output_ids);

        output = tokenizer
            .decode(&output_ids, true)
            .unwrap()
            .trim_start()
            .to_string();
        let first_space_or_newline = output
            .find(|c: char| c.is_whitespace() || c == '\n')
            .unwrap_or(output.len());

        let role = &output[..first_space_or_newline].trim_end().to_string();
        let out = &output[first_space_or_newline..].trim_start().to_string();

        let message_output = format!("{} {} \n {}\n{}", im_start, role, out, im_end);

        println!("{}", message_output);

        if input.trim() == "regenerate" {
            message_new = format!("{} \n {}", message, message_output);
            his_message_new = message;
        } else {
            message_new = format!(
                "{} \n {} \n {}",
                message,
                message_input.clone(),
                message_output
            );
            his_message_new = format!("{} \n {}", message, message_input.clone(),);
        }

        // let his_message_new = format!("{} \n {}", message, message_input.clone(),);

        chat_tmp.set_message(message_new);
        chat_tmp.set_his_message(his_message_new);
        chat_tmp.set_cache(new_cache);
        chat_tmp.set_his_cache(tmp_cache);
        chat_tmp.set_input_ids(tmp_ids);
        chat_tmp.set_re_input_ids(re_input_ids);
    }
}
