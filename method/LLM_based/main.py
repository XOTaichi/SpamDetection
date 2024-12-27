import csv
import json
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from method.LLM_based.detection.processor import preprocess_email, preprocessURLS, preprocessURLsPlainText
from method.LLM_based.detection.prompter import classify_email
from method.LLM_based import api
from method.LLM_based import local

# 设置日志记录
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(), 
                              logging.FileHandler('result/processing_llm.log', mode='w', encoding='utf-8')])

def process_line(i, line, model, fieldnames):
    """ 处理每一行数据的函数 """
    try:
        # 读取 JSONL 文件的每一行
        row = json.loads(line)
        mail = row["text"]
        actual_label = row["label_text"]

        # 假设邮件中包含一个 URL 列表，如果存在则进行处理
        url_info = None
        # if ENRICH_URL and 'urls' in row and len(row["urls"]) > 0:
        #     logging.info("Retrieving additional URL information...")
        #     url_to_analyze = row["urls"][0]  # 假设我们只分析第一个 URL
        #     # 如果您有URL分析器, 调用该部分
        #     # url_info = url_enricher.get_url_info(url_to_analyze)
        # else:
        #     logging.info("No URLs to enrich or URL enrichment is disabled.")

        # 调用模型进行分类
        classification_response, warning_msg = classify_email(email_input=mail, url_info=url_info, model=model)
        logging.info(f"Line {i} processed")
        logging.info(f"Actual label: {actual_label}")
        logging.info(f"Classification response: {classification_response}")
        logging.info(f"Warning message: {warning_msg}")

        # 将预测结果写入行
        row["predict"] = classification_response
        
        # 只写入所需字段到输出 JSONL
        filtered_row = {key: row[key] for key in fieldnames}
        return filtered_row

    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Error processing line {i}: {e}")
        return None  # 返回 None 表示该行处理失败

def main():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--model_type', type=str, choices=['local','api'], required=True, help='model type')
    parser.add_argument('--api_key', type=str, default='your_api_key', help='api key')
    parser.add_argument('--base_url', type=str, default='https://dashscope.aliyuncs.com/compatible-mode/v1', help='base url')
    parser.add_argument('--model_name', type=str, default="qwen-plus", help='model name')
    parser.add_argument('--model_path', type=str, help='model path')
    args = parser.parse_args()
    # 是否设定专门处理URL的部分
    ENRICH_URL = True

    # 加载模型
    if args.model_type == 'api':
        LLM = api.API_model(model_name=args.model_name, api_key=args.api_key, base_url=args.base_url)
    elif args.model_type == 'local':
        LLM = local.Local_model(model_path=args.model_path, model_name=args.model_name)
    else:
        raise ValueError("Invalid model type")
    #LLM = api.API_model(model_name="qwen-plus", api_key="", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    input_jsonl = r"dataset\body\enorn\test_data.jsonl"
    output_jsonl = r"result\body\llm_based\test_data_with_predictions.jsonl"
    
    fieldnames = ["message_id", "label_text", "predict"]

    # 创建线程池执行并行任务
    with open(input_jsonl, "r", encoding="utf-8") as infile, \
         open(output_jsonl, "w", newline="", encoding="utf-8") as outfile:

        # 初始化写入器
        outfile.write('')  # 清空文件内容
        
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = []
            i = 1
            for line in infile:
                logging.info(f"Task {i} started")
                futures.append(executor.submit(process_line, i, line, LLM, fieldnames))
                i += 1

            # 获取任务结果
            for future in as_completed(futures):
                result = future.result()
                if result:
                    # 将每个任务的结果写入 JSONL 文件
                    json.dump(result, outfile, ensure_ascii=False)
                    outfile.write("\n")  # 每个 JSON 对象占一行
                
        logging.info("Processing completed.")

if __name__ == "__main__":
    main()
