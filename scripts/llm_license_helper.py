import copy
import logging
import requests
import re
import yaml
import os
from datetime import datetime
from typing import Dict, Optional, List, Tuple

from api_license_fetcher import fetch_license_from_api

# 两阶段审计自动修复的最大重试次数
MAX_METADATA_RETRY = 3
MAX_TERMS_RETRY = 3


class LLMLicenseHelper:
    """
    使用 LLM 对未知许可证进行结构化建模与审计的辅助类。
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat", api_base: Optional[str] = None,
                 github_token: Optional[str] = None):
        # 支持的模型类型 (DeepSeek, 通义千问, OpenRouter)
        self.supported_models = {
            "deepseek": {
                "model": "deepseek-chat",
                "api_base": "https://api.deepseek.com/v1",
            },
            "qianwen": {
                "model": "qwen3-max",
                "api_base": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            },
            "openrouter": {
                "model": "google/gemini-2.5-pro",
                "api_base": "https://openrouter.ai/api/v1",
            },
        }
        self.github_token = github_token

        if model.lower() in self.supported_models:
            self.model_type = model.lower()
            self.model = self.supported_models[self.model_type]["model"]
            self.api_base = api_base or self.supported_models[self.model_type]["api_base"]
        else:
            self.model_type = "custom"
            self.model = model
            self.api_base = api_base or "https://api.deepseek.com/v1"

        self.api_key = api_key
        self.license_cache: Dict[str, str] = {}

        # 保存 LLM 输出的目录（用于调试与审计）
        self.llm_output_dir = os.path.join(os.getcwd(), "llm_outputs")
        os.makedirs(self.llm_output_dir, exist_ok=True)
        logging.info("LLM 输出将保存到: %s", self.llm_output_dir)

    # --------- 工具函数：保存输出 ---------

    def save_llm_output(self, license_name: str, llm_response: str, output_type: str = "license_analysis") -> str:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe_license_name = re.sub(r"[^a-zA-Z0-9._-]", "_", license_name)
            filename = f"{output_type}_{safe_license_name}_{timestamp}.txt"
            filepath = os.path.join(self.llm_output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# 许可证名称: {license_name}\n")
                f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 输出类型: {output_type}\n\n")
                f.write(llm_response)
            logging.info("LLM 输出已保存到: %s", filepath)
            return filepath
        except Exception as e:
            logging.error("保存 LLM 输出失败: %s", e)
            return ""

    def save_revision_for_inspection(
        self,
        license_name: str,
        data: Dict,
        revision_type: str,
        attempt: int,
        extra_label: str = "",
    ) -> str:
        """
        将每次自动修正后的内容保存为 YAML 文件，供人工检查。
        """
        try:
            revisions_dir = os.path.join(self.llm_output_dir, "revisions")
            os.makedirs(revisions_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", license_name)
            label = f"_{extra_label}" if extra_label else ""
            filename = f"{revision_type}_{safe_name}_attempt{attempt}{label}_{timestamp}.yaml"
            filepath = os.path.join(revisions_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# 许可证: {license_name}\n")
                f.write(f"# 修订类型: {revision_type}\n")
                f.write(f"# 第 {attempt} 次修正 | 保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(yaml.dump(data, allow_unicode=True, sort_keys=False))
            logging.info("修订内容已保存供检查: %s", filepath)
            print(f"修订内容已保存供检查: {filepath}")
            return filepath
        except Exception as e:
            logging.error("保存修订内容失败: %s", e)
            return ""

    # --------- 通用 LLM 调用 ---------

    def call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        timeout: int = 60,
    ) -> Optional[str]:
        if not self.api_key:
            logging.error("API key not provided for LLM license helper")
            return None

        try:
            if self.model_type == "qianwen":
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }
                data = {
                    "input": {"messages": messages},
                    "parameters": {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "result_format": "message",
                    },
                }
                response = requests.post(self.api_base, headers=headers, json=data, timeout=timeout)
            else:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }
                data = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                response = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=data, timeout=timeout)

            if response.status_code != 200:
                logging.error("LLM API 请求失败: %s - %s", response.status_code, response.text)
                return None

            data = response.json()
            if self.model_type == "qianwen":
                return (
                    data.get("output", {})
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logging.error("调用 LLM 失败: %s", e)
            return None

    # --------- 许可证文本获取 ---------

    def fetch_license_text(self, license_name: str) -> Optional[str]:
        if license_name in self.license_cache:
            return self.license_cache[license_name]

        urls = [
            f"https://raw.githubusercontent.com/spdx/license-list-data/main/text/{license_name}.txt",
            f"https://opensource.org/licenses/{license_name}.txt",
            f"https://www.gnu.org/licenses/{license_name}.txt",
        ]
        for url in urls:
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    text = resp.text.strip()
                    self.license_cache[license_name] = text
                    return text
            except Exception as e:
                logging.debug("Failed to fetch %s from %s: %s", license_name, url, e)

        # 本地 license_raw 目录（可选）
        try:
            base_dir = os.path.join(os.path.dirname(__file__), "license_raw")
            if os.path.exists(base_dir):
                license_files = [f for f in os.listdir(base_dir) if f.endswith(".txt")]
                processed_target = re.sub(r"[^a-zA-Z0-9]", "", license_name.lower())
                license_path = None

                for filename in license_files:
                    name = os.path.splitext(filename)[0]
                    processed = re.sub(r"[^a-zA-Z0-9]", "", name.lower())
                    if processed == processed_target:
                        license_path = os.path.join(base_dir, filename)
                        break

                if not license_path:
                    for filename in license_files:
                        name = os.path.splitext(filename)[0]
                        processed = re.sub(r"[^a-zA-Z0-9]", "", name.lower())
                        if processed_target in processed or processed in processed_target:
                            license_path = os.path.join(base_dir, filename)
                            break

                if license_path and os.path.exists(license_path):
                    with open(license_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        self.license_cache[license_name] = text
                        return text
        except Exception as e:
            logging.debug("Failed to read %s from local license_raw: %s", license_name, e)

        logging.warning("Could not fetch license text for %s", license_name)
        return None

    # --------- 元数据 & terms 生成---------

    def analyze_license_with_llm_data(self, license_name: str, license_text: str) -> Optional[Dict]:
        """使用 LLM 分析许可证并返回符合 licenses_description.yml 格式的元数据（不含 terms）。"""
        if not self.api_key:
            logging.error("API key not provided. Please set it when initializing LLMLicenseHelper")
            return None
        try:
            prompt = f"""
            你是一位专业的许可证分析专家，精通开源软件、开放数据、AI 模型及内容许可证的法律与技术细节。请严格依据以下规则分析给定的许可证：

            许可证名称: {license_name}
            许可证文本:
            {license_text}

            请仅输出符合 YAML 格式的许可证元数据，不要包含任何解释、注释、编号列表或中文说明。不要使用 Markdown 代码块（如 ```yaml）。若仅提供许可证名称而无文本，请基于权威知识（如 SPDX、OSI、Creative Commons 官方定义）填写；若有文本，则所有字段必须严格基于文本内容，不得臆测或补充未提及信息。
            输出 YAML 时，字段顺序必须严格按照本提示词出现的顺序
            full_name: 完整的许可证官方名称（不得添加用途说明如 "Data License"）
            short_id: 许可证的短标识符（必须与输入的 license_name 完全一致）
            url: 许可证的官方 URL；若不确定，请提供最合理的标准链接（如 SPDX 或官方站点）
            version: 许可证版本号；若无法确定，设为 null            
            available: true                                                

            categories: [public, software, data, model, proprietary, permissive, copyleft, disclose, auto-relicensing]
            # 必须从上述列表中选择适用项。注意            ：
            #   - 所有 Creative Commons、OSI 批准、SPDX 列表中的许可证均不得包含 "proprietary"            
            #   - "permissive"：宽松许可（如 MIT、Apache-2.0），根据实际内容判断，不得通过许可证名称简单臆测            
            #   - "copyleft"：具有传染性（如 GPL、CC-BY-SA）            
            #   - "disclose"：要求披露源码或修改（如 GPL 要求提供源代码）            
            #   - "auto-relicensing"：衍生作品必须以相同或兼容许可证发布（如 CC-BY-SA、GPL)            
            #   - "software"：此许可适用于software            
            #   - "data"：此许可适用于data            
            #   - "model"：此许可适用于model            
            #   - "proprietary"（专有）与 "permissive"（宽松）互斥，不能同时存在                        

            labels: []            
            # 可选标签（仅当明确适用时添加）：            
            #   "OSI Approved", "Public Domain", "GNU", "Creative Commons", "Open Data Commons",            
            #   "Microsoft", "OpenRAIL", "Responsible AI", "Use Restriction", "Meta"            

            rights: [use, modify, merge, redistribute, sublicense, commercial_use, patent_use]             
            # 仅列出许可证明确授予的权利（这里的权利是指不在分发状态下的权利。例如某许可证规定若修改则不允许分发，则rights中有modify而没有redistribute）仅使用上述值。注意：
            #   - "merge" 指将原作品与其他材料组合成集合，只要未修改原内容, 若许可证有授予类似含义的权利就可以添加(GPL-3.0允许merge)
            #   - 只能包含上述已列出内容，不得新加词汇                        
            #   - 重点："sublicense" 仅当许可证明文写出时才包含，不得臆测，若没有明文写出一律不得授予此权利
            #   - 再分发（redistribute）” 和 “再许可（sublicense）不同            
            #   - 针对CC许可证：如果许可证没有ND（例如CC-BY-NC-SA等），必须添加redistribute在rights中
            #   - 针对CC许可证：如果许可证要求ND（例如CC-BY-ND等），不得添加redistribute在rights中           
            #   - 针对CC许可证：都要有merge, modify在在rights中
            #   - 针对CC许可证：如果许可证不要求NC，则 rights 必须包含 commercial_use
            #   - 若许可证声明放弃版权（如 Unlicense、CC0），则添加 "copyright"
            #   - 不得包含未授权的行为(right和reserved_rights无交集,在right中出现的权利不得在reserved_rights中出现，反之亦然), 例如NC许可证禁止商业行为，即不得包含commercial_use
            #   - 在分析许可证时，请特别注意 patent_use 权利的状态, patent_use（专利使用权）指的是被许可人使用该专利技术的权利
            #   - 对于数据/内容类许可证（如 CC-BY、CC0、PDDL 等），不得包含 patent_use 在 rights 中
            #   - 对于模型/软件类许可证（如 Apache-2.0、MIT、OpenRAIL、OPT-175B 等），必须显式包含 patent_use 在 rights 中            

            reserved_rights: [trademark_use, copyright, patent, trademark, sublicense, commercial_use, patent_use, redistribute]
            # 仅列出许可证明确保留的法律权利，仅使用上述值。重要规则：
            #   - patent（专利权本身）指的是对某项发明所拥有的完整的专利所有权
            #   - patent_use（专利使用权）指的是被许可人使用该专利技术的权利
            #   - trademark（商标权本身）
            #   - trademark_use（商标使用权）指是否允许被许可人使用该商标
            #   - 只能包含上述列出内容，不得包含任何行为动 词（如 modify, copy, share, sell）
            #   - 例如许可证说 “you may not use for commercial purposes”，则 reserved_rights 必须包含 "commercial_use"
            #   - 若许可证说 “no right to sublicense”，则 reserved_rights 必须包含 "sublicense"
            #   - 所有 CC 默认保留：trademark, patent, sublicense（除非明示授予）
            #   - 针对CC许可证：如果许可证要求ND，有redistribute在reserved_rights
            #   - 对于数据/内容类许可证（如 CC-BY、CC0、PDDL 等），必须包含 patent_use 在 reserved_rights 中
            #   - right 和 reserved_rights 应无交集（disjoint），例如：
            #   - 针对CC许可证：如果许可证要求NC，则 rights 不得包含 commercial_use，reserved_rights 必须包含 commercial_use
            #   - 若许可证说类似 "no sublicensing"，则 rights 不得包含 sublicense，reserved_rights 必须包含 sublicense


             rights_prefix: [perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable, non-transferable, non-sublicensable, revocable, sublicensable, any-person]
            # 描述所授予权利的性质前缀，仅使用上述值。
            # - perpetual: 指授权一旦授予，在满足许可证条款的前提下永久有效，不会因时间到期而自动终止, 除非有明文写出（irrevocable不等同于perpetual），否则不得添加
            # - non-transferable: 被授权方不能将所获权利转授给第三方
            # - 注意区分：no-charge：排除任何形式的初始许可费，royalty-free：排除基于使用量、收入、分发数量等的后续分成或计费。

            coverage: [duplicate, derivative, modification, translation]
            # 表示哪些类型的作品可以被合法再分发：
            #   - "duplicate"：原样复制（除非许可证明确禁止复制或限制复制对象）
            #   - "derivative/modification/translation"：仅当许可证允许分发修改版时才可能包含
            #   - ND 类许可证（如 CC-BY-ND）只能包含 duplicate

            redistribute: [include_notice, include_license, state_changes, include_use_restriction, include_runtime_restriction]
            # 再分发时必须满足的要求。注意：
            #   - 只要涉及再分发，都必须满足这些条件
            #   - "state_changes" 仅当 coverage 包含 derivative/modification/translation 时出现才合理
            #   - ND 类许可证不得包含 state_changes
            #   - MIT许可证不得包含 state_changes
            #   - "include_use_restriction" 仅适用于明确附带使用限制的许可证

            compat: []
            # 仅当许可证**官方明确声明**与某许可证兼容时填写（如 CC-BY-SA 4.0 → GPL-3.0）

            incompat: []
            # 仅当许可证**官方明确声明**不兼容某些许可证时填写
            # OpenRAIL系列许可证（例如BigScience-BLOOM-RAIL-1.0）与GPL, AGPL, LGPL, GFDL这四种许可证不兼容，必须列出

            【最终校验清单】
            在输出前，请逐项确认：
            1. reserved_rights 中不含 modify/copy/share/sell 等行为动词。
            2. rights 中的 "sublicense" 仅在许可证明文授权时出现（CC 4.0、GPL、LGPL 均不包含）。
            3. rights = 被明确授予的使用权，reserved_rights = 明确未授予或禁止的权利，二者应无交集（disjoint）。在right中出现的权利不得在reserved_rights中出现，反之亦然
            4. 若 coverage 不包含 derivative/modification，则 redistribute 不得包含 state_changes。
            5. 所有 Creative Commons 许可证：
               - categories 必须包含 [public, data, (permissive|copyleft), auto-relicensing]
               - reserved_rights 必须包含 sublicense
               - rights 不得包含 sublicense
               - ND 类不得包含 modification/derivative in coverage
            6. 所有字段值必须来自预定义枚举
            """
            messages = [
                {"role": "system", "content": "你是一位专门分析软件和数据许可证的专家。"},
                {"role": "user", "content": prompt},
            ]
            llm_response = self.call_llm(messages=messages, temperature=0.3, max_tokens=2000)
            if not llm_response:
                return None
            self.save_llm_output(license_name, llm_response, "license_analysis_data")
            yaml_content = re.sub(r"^```(yaml)?\s*|\s*```$", "", llm_response.strip(), flags=re.MULTILINE)
            yaml_match = re.search(r"---\s*([\s\S]*?)\s*---", yaml_content)
            if yaml_match:
                yaml_content = yaml_match.group(1)
            return yaml.safe_load(yaml_content)
        except Exception as e:
            logging.error("Failed to analyze license with LLM: %s", e)
            return None

    def analyze_license_with_llm_terms(
        self, license_name: str, license_text: str, license_data_no_terms: str
    ) -> Optional[Dict]:
        """在已有元数据基础上，使用 LLM 生成 terms 并返回完整 license_data。"""
        if not self.api_key:
            return None
        try:
            prompt = f"""
            你是一位专业的许可证分析专家，精通开源软件、开放数据、AI 模型及内容许可证的法律与技术细节。你希望能在AI活动和许可证文本间架起桥梁。

            许可证名称: {license_name}
            许可证文本:
            {license_text}
            许可证元数据:
            {license_data_no_terms}
            # 任务：作为许可证分析专家，你需要在许可证元数据的基础上生成terms条目
            请严格依据以下规则分析给定的许可证文本
            请仅输出符合 YAML 格式的增加了terms的许可证元数据，不要包含任何解释、注释、编号列表或中文说明。不要使用 Markdown 代码块（如 ```yaml）。若仅提供许可证名称而无文本，请基于权威知识（如 SPDX、OSI、Creative Commons 官方定义）填写；若有文本，则所有字段必须基于文本内容。
            输出内容必须是完整的，包括**许可证元数据**和**terms**

            ## 分析核心原则：
            1.  **首要原则**：所有分析的最终依据必须是许可证文本本身。任何结论都应有文本措辞作为支撑或合理推断的起点。
            2.  **桥梁原则**：当许可证文本对AI相关活动（如训练、生成、微调、蒸馏、组合等）的法律性质**定义模糊或未定义**时，应使用下述分类法作为推断框架，将AI活动映射到许可证已有定义的相近概念上，以确定其`result`（结果形式）和可能触发的条款，将许可证对已有概念的规定应用在其对应的AI活动上。

            ## AI活动ModelGo分类法与许可证语言映射规则
            请依据以下分类处理`terms`字段中的`usages`：
            - **Combination (强分离组合, 原作品在新作品中可识别)**：
            - **AI活动示例**：模型集成（MoE）、Voting、Stacking
            - **可映射的许可证关键词**：`aggregate`, `collection`, `portion`, `link`, `use`, `arrange`, `separable`, `interface`
            - **Amalgamation (弱分离组合, 原作品难以分离)**：
            - **AI活动示例**：微调 (`fine-tune`)、model averaging、修改 (`modify`)、mix up、embed
            - **可映射的许可证关键词**：`modify`, `adapt`, `alter`, `remix`, `incorporate`, `revision`、`translate`
            - **Distillation (概念衍生, 新作品源于原作品的概念或功能)**：
            - **AI活动示例**：knowledge distillation、contrastive learning、S-T learning。
            - **可映射的许可证关键词**：transfer、reproduce、reuse
            - **Generation (数据衍生)**：
            - **AI活动示例**：Inference、Synthetic。
            - **可映射的许可证关键词**：`output`, `result`,
            - **如果许可证文本未提及任何AI活动以及可以映射的近似概念，那么相关AI活动`result`设置为`NODEF`          


            # terms 编写详细规则：
            - usages: []
                # 重点注意：先查看许可证元数据rights内容！确保该activity所需的所有rights均已在许可证元数据的rights字段中。
                # activity所需许可证元数据rights对照，活动：[所需rights]，如果缺失必要right，则不应包含此activity。(usages仅使用下述列举活动）
                # 举例：sell: [redistribute, sublicense, commercial_use], 当许可证right（许可证明确授予的权利）中有redistribute, sublicense, commercial_use(三个都要有)，才讨论sell在usages
                # 以下以此类推
                #copy: [use]  
                #use: [use]
                #share: [redistribute, sublicense]
                #sell: [redistribute, sublicense, commercial_use]
                #modify: [modify]
                #train: [use, modify]
                #combine: [use, merge]  #Combine multiple works to constrct a new work, but Recursive mixworks are not be included
                #combine_mix: [use, merge] #Combine multiple works to constrct a new work, the resulting work is a combination with mix types
                #amalgamate: [use, modify] #将多个作品进行融合，生成无法再分离的新作品
                #distill: [use]  # Distill knowledge from old models to new models
                #generate: [use]
                #embed: [use, modify]  # Embed works (corpus, image or other data samples) using aux_works (model or algorithm)
                #stat: [use]

            forms: [raw, binary, saas]
                # 三选一或多选：
                # - raw: 原始格式（源码、文本、数据文件）
                # - binary: 编译/打包后的可执行形式
                # - saas: 通过网络服务提供（如API、在线应用）

            result: 
                # 合法值：duplicate, independent, derivative, modification, translation, NODEF
                # CC许可证中embed活动的result应为translation
                # GPL-3.0许可证combine活动的forms是binary那么result是independent，如果combine活动的forms是raw那么result是derivative
                # 除以上情况基于以下逻辑选择一个：
                # 1. 如果许可证文本明确定义此activity的结果 → 使用文本定义
                #   -duplicate：仅限完全未改变原作品的行为
                #   -independent: 表示新作品与原作品在法律上是独立的，即使它们被组合在一起（例如通过链接、聚合等方式），也不构成对原作品的修改或衍生。新作品不受原许可证的传染性条款约束
                #   -derivative：表示新作品是原作品的衍生作品，即基于原作品进行了融合、改编等，新作品与原作品无法分离。需遵守原许可证的衍生作品条款，例如 GPL 要求整个衍生作品也必须以 GPL 发布（copyleft 效应）
                #   -modification：强调对原作品进行了内容层面的改动，比 derivative 更具体地指向“修改”行为本身，同样触发衍生作品义务
                #   -translation：特指对原作品进行语言或格式上的转换（如文本翻译、代码转写）
                #   -NODEF：处理许可证未覆盖的行为
                # 2. 否则，应用ModelGo分类法, 将AI活动映射到许可证已有定义的相近概念上，以确定其`result`（结果形式）和可能触发的条款
                #    - Generation (数据衍生) → 
                #        * 如果许可证将"输出"纳入管辖 → derivative
                #        * 否则 → NODEF
                # 3. 如果无法通过以上方式确定 → NODEF 


            restrictions: []
                # 适用的分发要求
                # 这里的restrictions 是针对特定 usage-result 组合的要求
                # 合法值（可多选，但不得出现未列举的值）：include_notice, include_license, state_changes, include_use_restriction, include_runtime_restriction
                # OpenRAIL类许可证（例如BigScience-BLOOM-RAIL-1.0，CreativeML-OpenRAIL-M等）terms中的restrictions必须至少有include_use_restriction和include_runtime_restriction            

            keywords: []
                # 从许可证文本中提取与此activity相关的关键词。keywords 必须直接来源于许可证文本中的原词或其词形变化，不得引入文本中不存在的概念
                # 示例：
                # - modify → [modify, adapt, alteration, derivative works]
                # - generate → [output, result, generation]
                # - combine → [portion, collection, aggregate]
                # - train → [train, fine-tune, improve] 

            relicense: false
                # 定义：针对该 usage 产生的结果（Result），用户是否拥有“再许可权”（即：将结果更换为不同于原作品的许可证进行发布）。
                # 请根据以下逻辑三选一：
                # - true: （允许更换 / 独立作品）
                #     1. 结果被视为“独立作品” (Independent)，不受原协议约束。
                #     2. 原协议是宽松协议 (Permissive, 如 MIT/Apache)，允许将衍生作品闭源或更换为私有协议（只要保留版权声明）。
                # - false: （禁止更换 / 强制继承）
                #     1. 强 Copyleft 协议的衍生作品：必须严格沿用原协议（传染性）。
                #     2. 专有协议 (Proprietary) 或禁止分发的协议：不允许更改协议条款，也不允许转为开源。
                # - conditional: （有条件更换）
                #     允许更换协议，但必须满足特定约束条件：
                #     1. 必须包含特定限制条款 (Use Restrictions)：如 Llama2/OpenRAIL，允许你用自己的协议发布微调模型，但必须保留“禁止用于暴力/非法用途”的条款。
                #     2. 必须在兼容列表内选择：只能转换为兼容的协议。
                # GPL-3.0许可证combine中如果combine活动的forms是raw那么relicense是conditional
                # OpenRAIL类许可证（例如BigScience-BLOOM-RAIL-1.0）的usages除了result是duplicate时除外，其余情况relicense必须为conditional
                # CC类许可证的usages是amalgamate时，relicense必须为conditional

                copyleft: false
                    # 布尔值：
                    # - true: 仅适用于具有"传染性"的许可证（如GPL、CC-BY-SA），要求衍生作品必须以相同许可证发布
                    # - false: 所有其他许可证              

                输出示例（仅示范格式）:
                  full_name: Educational Community License v2.0 # Software License
                  short_id: ECL-2.0
                  url: https://opensource.org/license/ecl-2-0/
                  version: 2.0
                  available: true
                  categories: [public, software, permissive]
                  labels: [OSI Approved]
                  rights: [use, modify, merge, redistribute, sublicense, patent_use, commercial_use]
                  reserved_rights: [trademark_use, copyright, patent, trademark]
                  rights_prefix: [perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable]
                  coverage: [duplicate, derivative]
                  redistribute: [include_notice, include_license]
                  terms:
                    - usages: [use, copy]
                      forms: [raw, binary]
                      result: duplicate
                      restrictions: []
                      relicense: false
                    - usages: [combine]
                      forms: [raw, binary]
                      result: independent
                      restrictions: []
                      keywords: [separable, link]
                      relicense: true
                    - usages: [amalgamate, modify]
                      forms: [raw, binary]
                      result: derivative
                      restrictions: [state_changes]
                      keywords: [modify, reproduction]
                      relicense: true
                    - usages: [train, distill, generate, embed]
                      forms: [raw, binary]
                      result: NODEF

                【最终校验清单】
                    在输出前，请逐项确认：
                    1. terms 中 result 必须与 usage 语义匹配。
                    2. 输出内容必须是完整的，包括**许可证元数据**和**terms**
                    3. copyleft 为 true 仅限 GPL、AGPL、LGPL、CC-BY-SA、CC-BY-NC-SA 等真正具有传染性的许可证。
                    4. 所有字段值必须来自预定义枚举。
                """
            messages = [
                {"role": "system", "content": "你是一位专门分析软件和数据许可证的专家。"},
                {"role": "user", "content": prompt},
            ]
            llm_response = self.call_llm(messages=messages, temperature=0.3, max_tokens=2000)
            if not llm_response:
                return None
            self.save_llm_output(license_name, llm_response, "license_analysis_terms")
            yaml_content = re.sub(r"^```(yaml)?\s*|\s*```$", "", llm_response.strip(), flags=re.MULTILINE)
            yaml_match = re.search(r"---\s*([\s\S]*?)\s*---", yaml_content)
            if yaml_match:
                yaml_content = yaml_match.group(1)
            return yaml.safe_load(yaml_content)
        except Exception as e:
            logging.error("Failed to analyze license terms with LLM: %s", e)
            return None

    def evaluate_metadata_compliance(self, license_name: str, license_text: str, metadata: Dict) -> Optional[Dict]:
        """
        阶段一审计：仅对基础元数据进行规则与语义一致性评估
        """
        if not self.api_key:
            logging.error("API key not provided.")
            return None

        try:
            metadata_yaml = yaml.dump(metadata, allow_unicode=True, sort_keys=False)

            prompt = f"""
            你是一位严谨的许可证结构化数据专家。
            你的任务是基于《合法枚举值字典》和《建模规则》，对输入的“基础元数据 YAML”进行编译级查错和语义校验。

            许可证名称: {license_name}
            许可证原文:
            {license_text}

            待复核的基础元数据 YAML:
            {metadata_yaml}

            【合法枚举值字典】
            categories 允许的值:
              - public
              - software
              - data
              - model
              - proprietary
              - permissive
              - copyleft
              - disclose
              - auto-relicensing

            rights 允许的值:
              - use
              - modify
              - merge
              - redistribute
              - sublicense
              - commercial_use
              - patent_use

            reserved_rights 允许的值:
              - trademark_use
              - copyright
              - patent
              - trademark  (注意：trademark 和 trademark_use 均合法，请勿误判！)
              - sublicense
              - commercial_use
              - patent_use
              - redistribute

            coverage 允许的值:
              - duplicate
              - derivative
              - modification
              - translation

            redistribute 允许的值:
              - include_notice
              - include_license
              - state_changes
              - include_use_restriction
              - include_runtime_restriction

            【校验规则】
            1. 枚举必须合法。
            2. 互斥性：rights 与 reserved_rights 必须完全无交集。
            3. 根据许可证内容检查每一项的值是否存在遗漏
            4. 联动：若 coverage 无 derivative/modification，则 redistribute 禁含 state_changes。
            3. CC规则：reserved_rights 必含 sublicense；rights 禁含 sublicense。

            【输出要求】（必须严格输出此 YAML，不含 Markdown 代码块）
            chain_of_thought: |
              步骤1：提取 YAML中的列表，逐一对比上方【合法枚举值字典】，确认是否都在字典内。
              步骤2：互斥性检查
              - rights 集合: [此处再次默写 rights]
              - reserved_rights 集合:[此处再次默写 reserved_rights]
              - 真实交集元素: [如有交集写出元素，如无写“无”]
              步骤3：结合文本分析语义是否进行了臆测, 是否有遗漏，是否有错误
              【针对易错点的特殊防御规则】（审查时必须绝对遵守，不得违背）：
                1. 商业使用一票否决权：只要许可证明确允许商业使用（无论是否有规模限制、行业限制、或者收费要求），`rights` 中必须包含 `commercial_use`，绝对不允许因为存在限制条款就判定 `commercial_use` 为错误。
                2. 再许可的严格定义：允许用户修改代码并以新协议发布衍生品，绝不等于拥有 `sublicense`（再许可）权！只要协议中明确出现了 "non-transferable"（不可转让）或没有出现明文的 "right to sublicense"，`sublicense` 权利就未被授予
                3. 归属声明的宽泛映射：当协议要求“保留版权声明”、“添加使用特定模型的说明”时，统一映射为 `redistribute: [include_notice]`。绝对禁止建议“添加自定义字段”！
                4. 软件和模型相关许可证中专利使用权的常识：只要协议允许“使用(use)”模型或软件，且未明文声明“不提供任何专利许可”，就不应该在 `reserved_rights` 中出现 `patent_use`（因为保留专利使用权意味着根本不能用）。
                5. 一个许可证不可能既是"proprietary"（专有）又是 "permissive"（宽松）的
                6. 根据许可证语义判断rights中是否应该有merge
            # 重要格式警告：
            # 1. 所有的 description 和 suggestion 字段的值，必须使用双引号 "" 完全包裹！
            # 2. 在自然语言描述中，绝对禁止使用英文冒号（:），请统一使用中文冒号（：）。
            ok: true/false
            usable_for_compliance: true/false
            rule_conformant: true/false
            semantic_conformant: true/false
            issues: 
              - type: rule | semantic
                field: 字段路径
                severity: critical | warning
                description: "用双引号包裹的中文简述问题"
                suggestion: "用双引号包裹的具体修改建议"
            summary: "中文整体结论"
            """

            messages = [
                {"role": "system",
                 "content": "你是一位严谨的许可证结构化数据专家"},
                {"role": "user", "content": prompt}
            ]

            llm_response = self.call_llm(
                messages=messages,
                temperature=0.1,
                max_tokens=2500
            )
            if not llm_response:
                return None

            # 保存评估报告
            self.save_llm_output(license_name, llm_response, "metadata_evaluation")

            # 解析为 YAML
            yaml_content = re.sub(r'^```(yaml)?\s*|\s*```$', '', llm_response.strip(), flags=re.MULTILINE)
            yaml_match = re.search(r'---\s*([\s\S]*?)(?:\s*---|$)', yaml_content)
            if yaml_match:
                yaml_content = yaml_match.group(1)
            evaluation_report = yaml.safe_load(yaml_content)
            if not isinstance(evaluation_report, dict):
                logging.error("Metadata evaluation report is not a dict")
                return None

            return evaluation_report

        except Exception as e:
            logging.error(f"Failed to evaluate metadata with LLM: {e}")
            return None

    def evaluate_terms_compliance(self, license_name: str, license_text: str, metadata_base: Dict, terms: List[Dict]) -> Optional[Dict]:
        """
        阶段二审计：在基础元数据通过的前提下，仅对 terms / ModelGo 映射部分进行规则与一致性评估。

        返回的评估报告为字典（从 YAML 解析而来），包含字段：
        - ok: bool，本阶段是否通过
        - usable_for_compliance: bool，terms 是否可用于合规性分析
        - rule_conformant: bool，是否符合 terms 建模规则
        - semantic_conformant: bool，是否与许可证原文及基础元数据含义一致
        - issues: 列表，逐条给出问题及修改建议（可包含 escalate_to_metadata 标记）
        - summary: 对整体情况的中文总结
        """
        if not self.api_key:
            logging.error("API key not provided. Please set it when initializing LLMLicenseHelper")
            return None

        try:
            metadata_yaml = yaml.dump(metadata_base, allow_unicode=True, sort_keys=False)
            terms_yaml = yaml.dump({"terms": terms}, allow_unicode=True, sort_keys=False)

            prompt = f"""
            你是一位专业的许可证分析与建模专家，精通开源软件、开放数据、AI 模型及内容许可证。
            下面给出：
            1）某许可证的原文，
            2）已经通过基础审计的元数据（不含 terms），
            3）在该元数据基础上生成的 terms / ModelGo 映射。
            请你只针对 terms 部分，从规则符合性、与元数据的一致性、与许可证原文的间接语义一致性三个角度进行审计。

            许可证名称:
            {license_name}

            许可证原文:
            {license_text}

            基础元数据 YAML（已通过基础审计，不含 terms）:
            {metadata_yaml}

            待审计的 terms YAML:
            {terms_yaml}

            一、规则维度（terms 建模规则）：
            1. usages 只能使用以下活动名：
               [use, copy, share, sell, modify, train, combine, combine_mix, amalgamate, distill, generate, embed, stat]
            2. forms 只能为 [raw, binary, saas] 中的一个或多个。
            3. result 只能为 [duplicate, independent, derivative, modification, translation, NODEF]。
               - CC 许可证中 embed 活动的 result 通常为 translation。
            4. restrictions 只能为 [include_notice, include_license, state_changes, include_use_restriction, include_runtime_restriction]。
            5. relicense 只能为 [true, false, conditional]。
               - OpenRAIL 类许可证的 usages 中，除 result 为 duplicate 的情况外，其余情况 relicense 通常应为 conditional。
               - CC 类许可证中 usages 为 amalgamate 时，relicense 通常应为 conditional。
            6. copyleft 只能为布尔值 true/false。

            二、与基础元数据的一致性：
            1. 所有 usages 必须满足对应的 rights 依赖关系（依赖关系示例）：
               - copy:    需要 [use]
               - use:     需要 [use]
               - share:   需要 [redistribute, sublicense]
               - sell:    需要 [redistribute, sublicense, commercial_use]
               - modify:  需要 [modify]
               - train:   需要 [use, modify]
               - combine: 需要 [use, merge]
               - combine_mix: 需要 [use, merge]
               - amalgamate: 需要 [use, modify]
               - distill: 需要 [use]
               - generate: 需要 [use]
               - embed:   需要 [use, modify]
               - stat:    需要 [use]
            2. 如果基础元数据的 rights 不满足某个 usage 所需的前置权利，而 terms 却包含了该 usage, 应当建议删除此usages

            三、与许可证原文的间接语义关系：
            1. 若许可证文本清晰规定了某类行为（例如衍生作品、输出、训练等）的法律后果，应优先按照文本定义检视 result/restrictions/relicense。
            2. 若文本未涉及 AI 行为及其近似概念，按照以下**AI活动ModelGo分类法与许可证语言映射规则**将AI活动映射为对应许可证语言进行检查：
            ## AI活动ModelGo分类法与许可证语言映射规则
            `terms`字段中的`usages`分类：
            - **Combination (强分离组合, 原作品在新作品中可识别)**：
            - **AI活动示例**：模型集成（MoE）、Voting、Stacking
            - **可映射的许可证关键词**：`aggregate`, `collection`, `portion`, `link`, `use`, `arrange`, `separable`, `interface`
            - **Amalgamation (弱分离组合, 原作品难以分离)**：
            - **AI活动示例**：微调 (`fine-tune`)、model averaging、修改 (`modify`)、mix up、embed
            - **可映射的许可证关键词**：`modify`, `adapt`, `alter`, `remix`, `incorporate`, `revision`、`translate`
            - **Distillation (概念衍生, 新作品源于原作品的概念或功能)**：
            - **AI活动示例**：knowledge distillation、contrastive learning、S-T learning。
            - **可映射的许可证关键词**：transfer、reproduce、reuse
            - **Generation (数据衍生)**：
            - **AI活动示例**：Inference、Synthetic。
            - **可映射的许可证关键词**：`output`, `result`,
            - **如果许可证文本未提及任何AI活动以及可以映射的近似概念，那么相关AI活动`result`设置为`NODEF`
            3. 注意relicense (再许可权/更换协议权) 定义：
               - true: 允许更换协议且无下游传染限制（适用于完全独立的衍生品，或 MIT/Apache 等宽松协议）。
               - false: 禁止更换协议，必须严格沿用原协议（适用于强 Copyleft 协议，或明确禁止更改协议的专有协议）。
               - conditional: 允许更换协议，但**必须满足原协议的特定强制约束条件**（例如保留“禁止非法用途”条款、保留“用户规模限制”、保留“特定归属声明”等）

            你的任务：
            1. 检查 terms 是否违反上述建模规则。
            2. 检查 terms 是否与基础元数据矛盾，是否需要删除。
            3. 检查 terms 是否在语义上明显偏离许可证文本（例如擅自推断许可/禁止某些 AI 行为）。
            4. 对每个问题给出分类和修改建议：

            输出要求（必须严格遵守）：
            - 只输出一段 YAML，不要包含任何解释性自然语言，不要使用 Markdown 代码块（如 ```yaml）。
            # 格式警告：所有 description 和 suggestion 必须用双引号 "" 包裹！绝对禁止在自然语言中使用英文冒号（:），请统一使用中文冒号（：）！
            - 输出 YAML 结构必须如下（可以根据需要在 issues 中增加多条记录）：
              chain_of_thought: |
              你必须按以下步骤：
              - 步骤1：建模规则检查
              - 步骤2：基础元数据的一致性检查
              - 步骤3：检查语义上是否明显偏离许可证文本
              ok: true/false
              usable_for_compliance: true/false
              rule_conformant: true/false
              semantic_conformant: true/false
              issues:
                - type: rule | semantic | mapping
                  field: 字段路径（例如 "terms[0].usages", "terms[1].result"）
                  severity: critical | warning
                  escalate_to_metadata: true/false
                  description: "用双引号包裹的中文问题描述"
                  suggestion: "用双引号包裹的中文修改建议"
              summary: "中文整体总结"
            """

            messages = [
                {"role": "system", "content": "你是一位严谨的许可证 terms / ModelGo 映射审计专家"},
                {"role": "user", "content": prompt},
            ]

            llm_response = self.call_llm(
                messages=messages,
                temperature=0.1,
                max_tokens=2500,
            )
            if not llm_response:
                return None

            # 保存评估报告
            self.save_llm_output(license_name, llm_response, "terms_evaluation")

            # 解析为 YAML
            yaml_content = re.sub(r'^```(yaml)?\s*|\s*```$', '', llm_response.strip(), flags=re.MULTILINE)
            yaml_match = re.search(r'---\s*([\s\S]*?)(?:\s*---|$)', yaml_content)
            if yaml_match:
                yaml_content = yaml_match.group(1)
            evaluation_report = yaml.safe_load(yaml_content)
            if not isinstance(evaluation_report, dict):
                logging.error("Terms evaluation report is not a dict")
                return None
            return evaluation_report

        except Exception as e:
            logging.error(f"Failed to evaluate terms with LLM: {e}")
            return None

    def _metadata_audit_passed(self, report: Optional[Dict]) -> bool:
        """阶段一审计是否通过：ok 且 usable_for_compliance 为 True，且无 critical 问题。"""
        if not report:
            return False
        if report.get("ok") is not True or report.get("usable_for_compliance") is not True:
            return False
        issues = report.get("issues") or []
        return not any(
            (isinstance(i, dict) and i.get("severity") == "critical") for i in issues
        )

    def _terms_audit_passed(self, report: Optional[Dict]) -> bool:
        """阶段二审计是否通过：ok 且 usable_for_compliance 为 True，且无 critical 问题。"""
        if not report:
            return False
        if report.get("ok") is not True or report.get("usable_for_compliance") is not True:
            return False
        issues = report.get("issues") or []
        return not any(
            (isinstance(i, dict) and i.get("severity") == "critical") for i in issues
        )

    def _has_escalate_to_metadata(self, report: Optional[Dict]) -> bool:
        """阶段二审计中是否存在需要回退到元数据层的问题（critical 且 escalate_to_metadata）。"""
        if not report:
            return False
        issues = report.get("issues") or []
        for i in issues:
            if not isinstance(i, dict):
                continue
            if i.get("severity") == "critical" and i.get("escalate_to_metadata") is True:
                return True
        return False

    def fix_metadata_with_llm(
        self,
        license_name: str,
        license_text: str,
        metadata: Dict,
        report: Dict,
        attempt: int = 1,
    ) -> Optional[Dict]:
        """
        根据阶段一审计报告，调用 LLM 修正基础元数据（不含 terms）。
        返回修正后的 metadata 字典，失败返回 None。修正后的内容会保存到 llm_outputs/revisions/ 供检查。
        """
        if not self.api_key:
            return None
        try:
            metadata_yaml = yaml.dump(metadata, allow_unicode=True, sort_keys=False)
            report_yaml = yaml.dump(report, allow_unicode=True, sort_keys=False)

            prompt = f"""
你是一位严谨的许可证结构化数据专家。请根据下方的「审计报告」对「当前基础元数据」进行修正，输出**且仅输出**修正后的完整基础元数据 YAML（不包含 terms 字段）。

要求：
1. 严格依据审计报告中的 issues 逐条修改，不得遗漏，不得擅自修改报告中未提及的问题。
2. 不得修改格式
4. 不要输出任何解释、Markdown 代码块或 chain_of_thought，只输出 YAML。
            
许可证名称: {license_name}
            
审计报告:
{report_yaml}
            
当前基础元数据（待修正）:
{metadata_yaml}
            
请直接输出修正后的完整基础元数据 YAML（字段顺序保持与当前元数据一致，不包含 terms）。
"""

            messages = [
                {"role": "system", "content": "你输出且仅输出合规的 YAML，无解释、无 Markdown 标记。"},
                {"role": "user", "content": prompt},
            ]
            llm_response = self.call_llm(
                messages=messages,
                temperature=0.1,
                max_tokens=2500,
            )
            if not llm_response:
                return None

            self.save_llm_output(license_name, llm_response, "license_metadata_fix")
            yaml_content = re.sub(r'^```(yaml)?\s*|\s*```$', '', llm_response.strip(), flags=re.MULTILINE)
            yaml_match = re.search(r'---\s*([\s\S]*?)(?:\s*---|$)', yaml_content)
            if yaml_match:
                yaml_content = yaml_match.group(1)
            fixed = yaml.safe_load(yaml_content)
            if isinstance(fixed, dict) and "terms" in fixed:
                del fixed["terms"]
            if fixed:
                self.save_revision_for_inspection(
                    license_name, fixed, "metadata_fix", attempt
                )
            return fixed
        except Exception as e:
            logging.error(f"fix_metadata_with_llm failed: {e}")
            return None

    def fix_terms_with_llm(
        self,
        license_name: str,
        license_text: str,
        metadata_base: Dict,
        terms: List[Dict],
        report: Dict,
        attempt: int = 1,
    ) -> Optional[List[Dict]]:
        """
        根据阶段二审计报告，调用 LLM 修正 terms（仅修改 terms 列表，不修改元数据）。
        返回修正后的 terms 列表，失败返回 None。修正后的内容会保存到 llm_outputs/revisions/ 供检查。
        """
        if not self.api_key:
            return None
        try:
            terms_yaml = yaml.dump({"terms": terms}, allow_unicode=True, sort_keys=False)
            report_yaml = yaml.dump(report, allow_unicode=True, sort_keys=False)
            metadata_yaml = yaml.dump(metadata_base, allow_unicode=True, sort_keys=False)

            prompt = f"""
你是一位专业的许可证 terms / ModelGo 映射专家。请根据下方的「审计报告」对「当前 terms」进行修正，输出**且仅输出**修正后的 terms 的 YAML（即一个名为 terms 的列表，结构为 terms: [...]）。
            
要求：
1. 严格依据审计报告中的 issues 逐条修改，不得遗漏，不得擅自修改报告中未提及的问题。
2. 不要输出任何解释、Markdown 代码块或 chain_of_thought，只输出 YAML。
            
许可证名称: {license_name}
            
已锁定的基础元数据:
{metadata_yaml}
            
审计报告:
{report_yaml}
            
当前 terms（待修正）:
{terms_yaml}
            
请直接输出修正后的 terms YAML，格式为 terms: 后跟列表。
            """

            messages = [
                {"role": "system", "content": "你输出且仅输出合规的 YAML，无解释、无 Markdown 标记。"},
                {"role": "user", "content": prompt},
            ]
            llm_response = self.call_llm(
                messages=messages,
                temperature=0.1,
                max_tokens=2500,
            )
            if not llm_response:
                return None

            self.save_llm_output(license_name, llm_response, "license_terms_fix")
            yaml_content = re.sub(r'^```(yaml)?\s*|\s*```$', '', llm_response.strip(), flags=re.MULTILINE)
            yaml_match = re.search(r'---\s*([\s\S]*?)(?:\s*---|$)', yaml_content)
            if yaml_match:
                yaml_content = yaml_match.group(1)
            fixed_block = yaml.safe_load(yaml_content)
            if isinstance(fixed_block, dict):
                fixed_terms = fixed_block.get("terms") or []
            else:
                fixed_terms = fixed_block or []
            if fixed_terms:
                self.save_revision_for_inspection(
                    license_name, {"terms": fixed_terms}, "terms_fix", attempt
                )
            return fixed_terms
        except Exception as e:
            logging.error(f"fix_terms_with_llm failed: {e}")
            return None

    def _default_terms(self) -> List[Dict]:
        """返回最小可用 terms 列表，供 get_license_data 在 LLM 未返回 terms 时使用。"""
        return [
            {
                "usages": ["use", "copy", "share", "combine", "generate", "train", "embed", "distill", "amalgamate", "modify"],
                "forms": ["raw", "binary", "saas"],
                "result": "NODEF",
                "restrictions": [],
                "relicense": True,
            }
        ]

    def get_license_data(self, license_name: str) -> Optional[Dict]:
        """
        获取未知许可证的数据。执行两阶段审计与自动修复循环：
        1. 生成基础元数据 -> 阶段一审计 -> 不通过则 LLM 修正并重试，直到通过或达到最大重试次数。
        2. 元数据锁定后生成 terms -> 阶段二审计 -> 不通过则仅修正 terms 并重试；若存在 escalate_to_metadata 的 critical 问题则停止并记录。
        3. 合并并返回最终 license_data。
        """
        # 1. 获取许可证文本
        license_text = self.fetch_license_text(license_name)
        if not license_text:
            logging.info("未获取到许可证全文，将仅基于名称由 LLM 推断元数据: %s", license_name)
            license_text = f"{license_name}"

        # 2. 生成基础元数据
        metadata_base = self.analyze_license_with_llm_data(license_name, license_text)
        if not metadata_base:
            return None

        # 3. 阶段一审计 + 自动修复循环
        metadata_evaluation = None
        for attempt in range(MAX_METADATA_RETRY):
            metadata_evaluation = self.evaluate_metadata_compliance(
                license_name, license_text, metadata_base
            )
            if self._metadata_audit_passed(metadata_evaluation):
                break
            if attempt < MAX_METADATA_RETRY - 1 and metadata_evaluation:
                logging.info(
                    f"基础元数据未通过阶段一审计，第 {attempt + 1} 次自动修正（许可证: {license_name}）。"
                )
                fixed = self.fix_metadata_with_llm(
                    license_name, license_text, metadata_base, metadata_evaluation,
                    attempt=attempt + 1,
                )
                if fixed:
                    metadata_base = fixed
                else:
                    logging.warning("LLM 修正基础元数据失败，保留当前版本并继续。")
                    break
            else:
                logging.warning(
                    f"基础元数据在 {MAX_METADATA_RETRY} 次内未通过阶段一审计，请根据评估报告人工修正（许可证: {license_name}）。"
                )
                break

        # 4. 元数据锁定，生成包含 terms 的完整数据
        metadata_base_locked = copy.deepcopy(metadata_base)
        self.save_revision_for_inspection(
            license_name, metadata_base_locked, "metadata_locked", 0
        )
        metadata_base_str = yaml.dump(metadata_base_locked, allow_unicode=True, sort_keys=False)
        license_data = self.analyze_license_with_llm_terms(
            license_name, license_text, metadata_base_str
        )
        if not license_data:
            return None

        terms = license_data.get("terms", []) or []

        # 5. 阶段二审计 + 自动修复循环（仅修改 terms）
        if terms:
            terms_evaluation = None
            for attempt in range(MAX_TERMS_RETRY):
                terms_evaluation = self.evaluate_terms_compliance(
                    license_name=license_name,
                    license_text=license_text,
                    metadata_base=metadata_base_locked,
                    terms=terms,
                )
                if self._terms_audit_passed(terms_evaluation):
                    break
                if self._has_escalate_to_metadata(terms_evaluation):
                    logging.warning(
                        f"terms 审计发现需回退到元数据层的问题，停止 terms 自动修正（许可证: {license_name}）。请先修正基础元数据后重新生成 terms。"
                    )
                    break
                if attempt < MAX_TERMS_RETRY - 1 and terms_evaluation:
                    logging.info(
                        f"terms 未通过阶段二审计，第 {attempt + 1} 次自动修正（许可证: {license_name}）。"
                    )
                    fixed_terms = self.fix_terms_with_llm(
                        license_name,
                        license_text,
                        metadata_base_locked,
                        terms,
                        terms_evaluation,
                        attempt=attempt + 1,
                    )
                    if fixed_terms:
                        terms = fixed_terms
                        license_data["terms"] = terms
                    else:
                        logging.warning("LLM 修正 terms 失败，保留当前版本并继续。")
                        break
                else:
                    logging.warning(
                        f"terms 在 {MAX_TERMS_RETRY} 次内未通过阶段二审计，请根据评估报告人工修正（许可证: {license_name}）。"
                    )
                    break

        # 6. 验证和完善最终数据
        if "short_id" not in license_data:
            license_data["short_id"] = license_name
        required_fields = ["full_name", "short_id", "url", "categories", "rights", "reserved_rights", "terms"]
        for field in required_fields:
            if field not in license_data:
                logging.warning(f"Missing required field {field} in license data")
                if field == "full_name":
                    license_data[field] = license_name
                elif field == "url":
                    license_data[field] = f"https://example.com/license/{license_name}"
                elif field in ["categories", "rights", "reserved_rights"]:
                    license_data[field] = []
                elif field == "terms":
                    license_data[field] = []

        # 7. 保存最终合并结果供检查
        self.save_revision_for_inspection(
            license_name, license_data, "license_final", 0
        )
        return license_data

    # --------- 未知许可证处理（去掉交互式 input）---------

    def handle_unknown_license(self, component_name: str) -> Tuple[str, Optional[str]]:
        """
        处理未知许可证的组件（无交互版本，适用于服务端场景）。
        尝试顺序：
        1）LLM 按组件名猜测许可证；
        2）GitHub/HF API 查询；
        3）从 README / LICENSE 文本中用 LLM 进一步识别；
        4）都失败则保守返回 "Unlicense"。
        """
        logging.info("组件 %s 许可证未知，将尝试自动查询。", component_name)
        final_name = component_name

        try:
            # 1. 使用 LLM 直接按组件名查询
            license_name = self.query_license_by_component_name(final_name)
            if license_name:
                logging.info("通过 LLM 查询到组件 %s 的许可证：%s", final_name, license_name)
                return license_name, None

            # 2. 尝试通过 API 查询（GitHub / Hugging Face）
            logging.info("LLM 未能直接确定许可证，尝试通过 API 查询...")
            api_result = fetch_license_from_api(final_name)
            if api_result:
                license_name, license_text, readme_text = api_result
                logging.info("API 查询成功：组件 %s 许可证为 %s", final_name, license_name)

                if license_name is None or (isinstance(license_name, str) and license_name.lower() == "other"):
                    logging.info(
                        "API 返回的许可证为 %s，将尝试从 README 或 LICENSE 内容中进一步分析。",
                        license_name,
                    )
                    analyzed_license = None
                    if readme_text:
                        analyzed_license = self.analyze_license_from_readme(final_name, readme_text)
                    if not analyzed_license and license_text:
                        analyzed_license = self.analyze_license_text_for_component(final_name, license_text)
                    if analyzed_license:
                        license_name = analyzed_license
                        logging.info("从内容中分析出许可证名称：%s", license_name)
                    else:
                        license_name = f"{component_name}_license"
                        logging.info("无法从内容中分析出许可证名称，使用占位名：%s", license_name)

                    if license_text or readme_text:
                        self.save_license_raw_content(license_name, license_text or readme_text)

                return license_name, None

            logging.warning(
                "API 查询失败：将设置该组件许可证为 Unlicense 以避免与其他组件产生冲突，结果仅供参考。"
            )
            return "Unlicense", None

        except Exception as e:
            logging.error("处理未知许可证时发生错误: %s", e)
            return "Unlicense", str(e)

    # --------- 其余辅助函数：与 ModelGo2-main 基本一致 ---------
    # query_license_by_component_name / analyze_license_text_for_component /
    # analyze_license_from_readme / save_license_raw_content ...

    def query_license_by_component_name(self, component_name: str) -> Optional[str]:
        if not self.api_key:
            return None
        try:
            prompt = f"""
            请诚实回答：你知道'{component_name}'这个组件的默认许可证是什么吗？
            如果你确实知道，请直接返回许可证名称，不要添加任何其他解释。
            如果你不知道或不确定，请直接回答'不知道'。
            """
            messages = [
                {"role": "system", "content": "你是一位软件和数据许可证专家，只回答你确定的事实。"},
                {"role": "user", "content": prompt},
            ]
            answer = self.call_llm(messages=messages, temperature=0.3, max_tokens=150)
            if not answer:
                return None
            self.save_llm_output(component_name, answer, "component_license_query")
            if answer.lower() in ["不知道", "不", "no", "don't know", "i don't know"]:
                return None
            return answer.strip()
        except Exception as e:
            logging.error("查询组件许可证时发生错误: %s", e)
            return None

    def analyze_license_text_for_component(self, component_name: str, license_text: str) -> Optional[str]:
        if not self.api_key:
            return None
        try:
            prompt = f"""
            请分析以下文本，确定'{component_name}'组件使用的许可证名称。
            
            文本内容：
            {license_text}
            
            请根据文本内容直接返回许可证名称，如果无法确定，请返回'Unknown'。
            """
            messages = [
                {"role": "system", "content": "你是一位许可证分析专家，能够从文本中识别标准许可证。"},
                {"role": "user", "content": prompt},
            ]
            license_name = self.call_llm(messages=messages, temperature=0.1, max_tokens=50)
            if not license_name:
                return None
            self.save_llm_output(component_name, license_name, "license_text_analysis")
            if license_name == "Unknown":
                return None
            return license_name
        except Exception as e:
            logging.error("分析许可证文本时发生错误: %s", e)
            return None

    def analyze_license_from_readme(self, component_name: str, readme_content: str) -> Optional[str]:
        if not self.api_key:
            return None
        try:
            prompt = f"""
            请从以下 README 内容中分析'{component_name}'组件使用的许可证名称。
            
            README 内容：
            {readme_content}
            
            请直接返回许可证名称，如果无法确定，请返回'Unknown'。
            """
            messages = [
                {"role": "system", "content": "你是一位许可证分析专家，能够从 README 中识别标准许可证。"},
                {"role": "user", "content": prompt},
            ]
            license_name = self.call_llm(messages=messages, temperature=0.1, max_tokens=50)
            if not license_name or license_name == "Unknown":
                return None
            return license_name
        except Exception as e:
            logging.error("Failed to analyze license from README for component %s: %s", component_name, e)
            return None

    def save_license_raw_content(self, license_name: str, license_content: str) -> str:
        base_dir = os.path.join(os.path.dirname(__file__), "license_raw")
        os.makedirs(base_dir, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9-_.]", "-", license_name)
        file_path = os.path.join(base_dir, f"{safe_name}.txt")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(license_content)
            logging.info("License content saved to: %s", file_path)
            print(f"许可证全文已保存到: {file_path}")
        except Exception as e:
            logging.error("Failed to save license content to %s: %s", file_path, e)
            print(f"保存许可证全文失败: {e}")
        return file_path


# 全局实例
llm_helper = LLMLicenseHelper()


def set_api_key(api_key: str, model: str = "deepseek", github_token: Optional[str] = None) -> LLMLicenseHelper:
    """
    设置全局 llm_helper 的 API key 和模型。
    """
    global llm_helper
    llm_helper.api_key = api_key

    supported_models = {
        "deepseek": {
            "model": "deepseek-chat",
            "api_base": "https://api.deepseek.com/v1",
        },
        "qianwen": {
            "model": "qwen-turbo",
            "api_base": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        },
        "openrouter": {
            "model": "google/gemini-2.5-pro",
            "api_base": "https://openrouter.ai/api/v1",
        },
    }

    if model.lower() in supported_models:
        llm_helper.model_type = model.lower()
        llm_helper.model = supported_models[model.lower()]["model"]
        llm_helper.api_base = supported_models[model.lower()]["api_base"]
    else:
        llm_helper.model_type = "custom"
        llm_helper.model = model

    llm_helper.github_token = github_token
    logging.info("%s API key has been configured for global llm_helper", model)
    print(f"已配置 {model} API 密钥用于许可证分析")
    return llm_helper

