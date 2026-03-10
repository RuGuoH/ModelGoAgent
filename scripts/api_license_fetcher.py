import os
import requests
import json
import logging
from typing import Optional, Tuple
from huggingface_hub import list_models

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class APILicenseFetcher:
    """API 许可证查询器，用于通过 API 获取组件的许可证信息。"""

    def __init__(self, huggingface_token: Optional[str] = None, github_token: Optional[str] = None):
        # 设置请求超时时间
        self.timeout = 10
        # 请求头
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "ModelGo-License-Fetcher/1.0",
        }
        # 优先使用传入的 Hugging Face / GitHub token，否则尝试从环境变量读取
        # 不再使用任何硬编码密钥
        self.huggingface_token = (
            huggingface_token if huggingface_token is not None else os.getenv("HUGGINGFACE_TOKEN")
        )
        self.github_token = github_token if github_token is not None else os.getenv("GITHUB_TOKEN")

    def fetch_github_license(self, github_url: str) -> Optional[Tuple[str, Optional[str]]]:
        """
        通过 GitHub API 获取仓库的许可证信息。

        Args:
            github_url: GitHub 仓库 URL，如 https://github.com/huggingface/transformers

        Returns:
            (许可证名称, 许可证全文)；若失败返回 None
        """
        try:
            if "github.com" not in github_url:
                return None

            parts = github_url.split("github.com/")[-1].strip("/").split("/")
            if len(parts) < 2:
                # 只给了仓库名，尝试通过搜索获取最匹配的仓库
                repo_name = parts[0]
                github_token = self.github_token
                headers = {
                    "Accept": "application/vnd.github.v3+json",
                }
                if github_token:
                    headers["Authorization"] = f"token {github_token}"

                search_url = "https://api.github.com/search/repositories"
                params = {
                    "q": f"{repo_name} in:name",
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 1,
                }

                resp = requests.get(search_url, headers=headers, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()

                if data.get("total_count", 0) == 0:
                    logger.warning("未找到名为 '%s' 的仓库", repo_name)
                    return None

                best_match = data["items"][0]
                full_name = best_match["full_name"]
                parts = full_name.strip("/").split("/")

            owner, repo = parts[0], parts[1]
            api_url = f"https://api.github.com/repos/{owner}/{repo}"
            logger.info("Fetching GitHub license info for %s/%s", owner, repo)

            gh_headers = self.headers.copy()
            if self.github_token:
                gh_headers["Authorization"] = f"token {self.github_token}"

            response = requests.get(api_url, headers=gh_headers, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if "license" in data and data["license"]:
                    license_name = data["license"].get("name")
                    license_spdx = data["license"].get("spdx_id")
                    logger.info("Found license: %s (%s)", license_name, license_spdx)

                    license_text = None
                    if license_spdx:
                        license_api_url = f"https://api.github.com/repos/{owner}/{repo}/license"
                        license_response = requests.get(license_api_url, headers=gh_headers, timeout=self.timeout)
                        if license_response.status_code == 200:
                            license_data = license_response.json()
                            content = license_data.get("content")
                            if content:
                                import base64

                                license_text = base64.b64decode(content).decode("utf-8")
                    return license_name, license_text

                # license 字段为空时，尝试直接访问 license 端点
                license_api_url = f"https://api.github.com/repos/{owner}/{repo}/license"
                license_response = requests.get(license_api_url, headers=gh_headers, timeout=self.timeout)
                if license_response.status_code == 200:
                    license_data = license_response.json()
                    if "license" in license_data and license_data["license"]:
                        license_name = license_data["license"].get("name")
                        license_spdx = license_data["license"].get("spdx_id")
                        logger.info("Found license via direct endpoint: %s (%s)", license_name, license_spdx)

                        content = license_data.get("content")
                        license_text = None
                        if content:
                            import base64

                            license_text = base64.b64decode(content).decode("utf-8")
                        return license_name, license_text

            else:
                logger.warning("GitHub API request failed with status: %s", response.status_code)
                if response.status_code == 403:
                    logger.warning("GitHub API rate limit exceeded or authentication required")

        except Exception as e:
            logger.error("Error fetching GitHub license: %s", str(e))

        return None

    def fetch_huggingface_license(self, hf_url_or_id: str) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
        """
        通过 Hugging Face API 获取模型或数据集的许可证信息。

        Args:
            hf_url_or_id: HF 模型/数据集 URL 或 ID，如 gpt2 或 https://huggingface.co/gpt2

        Returns:
            (许可证名称, 许可证全文, README 全文)；若失败返回 None
        """
        try:
            if "huggingface.co/" in hf_url_or_id:
                model_id = hf_url_or_id.split("huggingface.co/")[-1].strip("/")
            else:
                model_id = hf_url_or_id.strip("/")

            models = list_models(
                search=model_id,
                sort="downloads",
                direction=-1,
                limit=10,
            )

            if models:
                most_relevant_model_id = next(models).modelId
                logger.info("最相关的 Model ID 可能是: %s", most_relevant_model_id)
            else:
                logger.info("没有找到匹配的模型，使用原始 ID。")
                most_relevant_model_id = model_id

            model_ids_to_try = [most_relevant_model_id]

            for current_model_id in model_ids_to_try:
                api_url = f"https://huggingface.co/api/models/{current_model_id}"
                logger.info("Fetching Hugging Face license info for %s", current_model_id)

                hf_headers = self.headers.copy()
                if self.huggingface_token:
                    hf_headers["Authorization"] = f"Bearer {self.huggingface_token}"

                response = requests.get(api_url, headers=hf_headers, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    license_name = None

                    if data.get("cardData") and "license" in data["cardData"]:
                        license_name = data["cardData"]["license"]
                    logger.info("Found license in cardData: %s", license_name)

                    if not license_name and "tags" in data:
                        license_tags = [t for t in data["tags"] if t.startswith("license:")]
                        if license_tags:
                            license_name = license_tags[0].split(":", 1)[1]
                    logger.info("Found license in tags: %s", license_name)

                    if license_name:
                        license_text = self._fetch_huggingface_file(current_model_id, "LICENSE")
                        if not license_text and "/" in current_model_id:
                            try:
                                direct_url = f"https://huggingface.co/{current_model_id}/raw/main/LICENSE"
                                file_resp = requests.get(direct_url, headers=hf_headers, timeout=self.timeout)
                                if file_resp.status_code == 200:
                                    license_text = file_resp.text
                                    logger.info("Successfully fetched LICENSE via direct URL for %s", current_model_id)
                            except Exception as e:
                                logger.error("Error fetching LICENSE via direct URL: %s", str(e))

                        readme_text = self._fetch_huggingface_file(current_model_id, "README.md")
                        return license_name, license_text, readme_text
                else:
                    logger.debug(
                        "Hugging Face model API request failed with status: %s for %s",
                        response.status_code,
                        current_model_id,
                    )

            # 尝试数据集 API
            dataset_api_url = f"https://huggingface.co/api/datasets/{model_id}"
            hf_headers = self.headers.copy()
            if self.huggingface_token:
                hf_headers["Authorization"] = f"Bearer {self.huggingface_token}"

            dataset_response = requests.get(dataset_api_url, headers=hf_headers, timeout=self.timeout)
            if dataset_response.status_code == 200:
                dataset_data = dataset_response.json()
                if dataset_data.get("cardData") and "license" in dataset_data["cardData"]:
                    license_name = dataset_data["cardData"]["license"]
                    logger.info("Found license in dataset cardData: %s", license_name)
                    license_text = self._fetch_huggingface_file(model_id, "LICENSE", is_dataset=True)
                    readme_text = self._fetch_huggingface_file(model_id, "README.md", is_dataset=True)
                    return license_name, license_text, readme_text

            logger.warning("All Hugging Face API requests failed for %s", model_id)

        except Exception as e:
            logger.error("Error fetching Hugging Face license: %s", str(e))

        return None

    def detect_source_and_fetch_license(self, url_or_id: str) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
        """
        自动检测来源并获取许可证信息。

        Args:
            url_or_id: 组件 URL 或 ID

        Returns:
            (许可证名称, 许可证全文, README 全文)；若失败返回 None
        """
        if "github.com" in url_or_id:
            result = self.fetch_github_license(url_or_id)
            if result:
                name, text = result
                return name, text, None
            logger.warning("GitHub API failed to find license for %s", url_or_id)

        if "huggingface.co" in url_or_id or len(url_or_id.split("/")) <= 2:
            result = self.fetch_huggingface_license(url_or_id)
            if result:
                return result
            logger.warning("Hugging Face API failed to find license for %s", url_or_id)

        logger.info("Ambiguous source, trying both APIs for %s", url_or_id)

        # 先假设为 GitHub 仓库名
        result = self.fetch_github_license(f"https://github.com/{url_or_id}")
        if result:
            name, text = result
            return name, text, None

        # owner/repo 形式再尝试一次
        if len(url_or_id.split("/")) == 2 and "." not in url_or_id.split("/")[1]:
            result = self.fetch_github_license(f"https://github.com/{url_or_id}")
            if result:
                name, text = result
                return name, text, None

        # 再尝试 HF
        result = self.fetch_huggingface_license(url_or_id)
        if result:
            return result

        logger.warning("Cannot detect source for: %s and both APIs failed", url_or_id)
        return None

    def _fetch_huggingface_file(self, repo_id: str, filename: str, is_dataset: bool = False) -> Optional[str]:
        """
        从 Hugging Face 仓库获取指定文件内容。
        """
        try:
            endpoint = "datasets" if is_dataset else "models"
            api_url = f"https://huggingface.co/api/{endpoint}/{repo_id}/revision/main/tree/{filename}"

            hf_headers = self.headers.copy()
            if self.huggingface_token:
                hf_headers["Authorization"] = f"Bearer {self.huggingface_token}"

            response = requests.get(api_url, headers=hf_headers, timeout=self.timeout)
            if response.status_code == 200:
                file_data = response.json()
                if "url" in file_data:
                    file_response = requests.get(file_data["url"], headers=hf_headers, timeout=self.timeout)
                    if file_response.status_code == 200:
                        return file_response.text

            logger.debug("Failed to fetch %s from Hugging Face repo %s", filename, repo_id)
            return None

        except Exception as e:
            logger.error("Error fetching %s from Hugging Face repo %s: %s", filename, repo_id, str(e))
            return None


# 全局实例
api_license_fetcher = APILicenseFetcher()


def fetch_license_from_api(
    url_or_id: str, huggingface_token: Optional[str] = None, github_token: Optional[str] = None
) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
    """
    从 API 获取许可证信息的便捷函数。
    """
    if huggingface_token or github_token:
        fetcher = APILicenseFetcher(huggingface_token=huggingface_token, github_token=github_token)
        return fetcher.detect_source_and_fetch_license(url_or_id)
    return api_license_fetcher.detect_source_and_fetch_license(url_or_id)

