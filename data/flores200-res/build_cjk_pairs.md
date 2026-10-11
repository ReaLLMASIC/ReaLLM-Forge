请在当前 working directory 实现一个可复现的数据处理 pipeline。

输入文件：text_jpn_Jpan.txt, text_kor_Hang.txt, text_zho_Hans.txt
输出文件：data/cjk_sentence_pairs.json

目标：
根据输入文件下载数据；数据包含中文、日文、韩文三种语言。请抽取 sentence-level 的 CJK translation pairs，并输出 JSON。

强约束：
- CJK 语言代码统一为 zh、ja、ko。
- 只抽取数据中已有的平行句/对齐句，禁止用 LLM 或翻译 API 生成翻译。
- 输出 JSON 必须 UTF-8 保留 CJK 字符，不要 ASCII escape。
- 输出必须 deterministic：稳定排序、稳定 id、重复可检测。
- 只新增文件，不改现有输入文件
- 实现 CLI，例如：
  python -m <module> --input <INPUT_FILE_PATH> --out data/cjk_sentence_pairs.json

JSON schema：
{
    "source": "flores200-res",
    "languages": ["kor_Hang", "zho_Hans", "jpn_Jpan"],
    "records": [
      {
        "translations": {
          "kor_Hang": "...",
          "zho_Hans": "...",
          "jpn_Jpan": "..."
        }
      }
    ]
  }


pair 规则：
- 对每个 zh/ja/ko 三语对齐句
- 不输出 src_lang == tgt_lang。
- 不输出空句子。

请先 inspect repo 和输入文件，再实现。
实现后运行：
1. 生成 JSON 的命令
2. python -m json.tool data/cjk_sentence_pairs.json > /dev/null
3. 如果项目有测试，运行测试；如果没有，请添加最小测试。

最后汇报：
- 修改了哪些文件
- 如何复现
- 输出路径
- 生成了多少 sentence pairs
- 跳过了多少记录以及原因
- 做了哪些假设