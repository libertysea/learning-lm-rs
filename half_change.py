from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定模型目录  
model_dir = "./models/chat"

# 加载预训练的模型和分词器  
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 将模型转换为 float16（半精度）  
model = model.half()

# 创建一个新的目录来保存转换后的模型  
output_dir = model_dir + "_f16"

# 保存转换后的模型和分词器  
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"模型已成功保存为半精度格式到 {output_dir}")
