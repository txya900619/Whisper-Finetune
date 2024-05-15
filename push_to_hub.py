import argparse
import functools
import os

from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizerFast,\
    WhisperProcessor
from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("model_path", type=str, default="output/whisper-tiny/checkpoint-best/", help="微调保存的模型路径")
add_arg('output_dir', type=str, default='models/',    help="合并模型的保存目录")
add_arg("local_files_only", type=bool, default=False, help="是否只在本地加载模型，不尝试下载")
args = parser.parse_args()
print_arguments(args)

# 检查模型文件是否存在
assert os.path.exists(args.model_path), f"模型文件{args.model_path}不存在"
# 获取Lora配置参数
# 获取Whisper的基本模型
model = WhisperForConditionalGeneration.from_pretrained(args.model_path, device_map={"": "cpu"},
                                                             local_files_only=args.local_files_only)
model_base_id = "openai/whisper-large-v3"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_base_id,
                                                            local_files_only=args.local_files_only)
tokenizer = WhisperTokenizerFast.from_pretrained(model_base_id,
                                                 local_files_only=args.local_files_only)
processor = WhisperProcessor.from_pretrained(model_base_id,
                                             local_files_only=args.local_files_only)

model.train(False)



# 保存模型到指定目录中
model.push_to_hub("formospeech/whisper-large-v3-taiwanese-hakka")
feature_extractor.push_to_hub("formospeech/whisper-large-v3-taiwanese-hakka")
tokenizer.push_to_hub("formospeech/whisper-large-v3-taiwanese-hakka")
processor.push_to_hub("formospeech/whisper-large-v3-taiwanese-hakka")
