import json
import os
from typing import Dict, Any, Optional

import openai
import httpx
import pandas as pd
from openai import OpenAI


def format_analysis_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    格式化分析结果，确保输出格式统一

    Args:
        result (Dict[str, Any]): 原始分析结果

    Returns:
        Dict[str, Any]: 格式化后的结果
    """
    if not result:
        return {
            "AI分析结果": {
                "分析状态": "分析失败",
                "失败原因": "无法获取API响应",
                "技术分析": "数据获取失败，无法提供分析。",
                "走势分析": "数据获取失败，无法提供分析。",
                "投资建议": "由于数据获取失败，暂不提供投资建议。",
                "风险提示": "数据不完整，投资决策需谨慎。"
            }
        }

    return result


def _create_system_prompt() -> str:
    """
    创建系统提示词

    Returns:
        str: 系统提示词
    """
    return """你是一个专业的金融分析师，将收到完整的股票历史数据和技术指标数据进行分析。
数据包括：
1. 历史数据：每日的开盘价、收盘价、最高价、最低价和成交量
2. 技术指标：所有交易日的各项技术指标数据
3. 市场趋势：当前的关键趋势数据

请基于这些完整的历史数据进行深入分析，包括以下方面：

1. 技术面分析
- 通过历史数据分析长期趋势
- 识别关键的支撑位和压力位
- 分析重要的技术形态
- 对所有技术指标进行综合研判
- 寻找指标之间的背离现象

2. 走势研判
- 判断当前趋势的强度和可能持续性
- 识别可能的趋势转折点
- 分析成交量和价格的配合情况
- 预判可能的运行区间

3. 投资建议
- 基于完整数据给出明确的操作建议
- 设置合理的止损和目标价位
- 建议适当的持仓时间和仓位控制
- 针对不同投资周期给出建议

4. 风险提示
- 通过历史数据识别潜在风险
- 列出需要警惕的技术信号
- 提供风险规避的具体建议
- 说明需要持续关注的指标

请注意：
- 结合全部历史数据做出判断
- 分析结果要有数据支持
- 避免过度简化或主观判断
- 必要时引用具体的历史数据点
- 结合多个维度的指标进行交叉验证

按照以下固定格式输出分析结果,不要包含任何markdown标记：

技术分析
1. 长期趋势分析：
趋势判断
突破情况
形态分析

2. 支撑和压力：
关键支撑位
关键压力位
突破可能性

3. 技术指标研判：
MACD指标
KDJ指标
RSI指标
布林带分析
其他关键指标

走势分析
1. 当前趋势：
趋势方向
趋势强度
持续性分析

2. 价量配合：
成交量变化
量价关系
市场活跃度

3. 关键位置：
当前位置
突破机会
调整空间

投资建议
1. 操作策略：
总体建议
买卖时机
仓位控制

2. 具体参数：
止损位设置
目标价位
持仓周期

3. 分类建议：
激进投资者建议
稳健投资者建议
保守投资者建议

风险提示
1. 风险因素：
技术面风险
趋势风险
位置风险

2. 防范措施：
止损设置
仓位控制
注意事项

3. 持续关注：
重点指标
关键价位
市场变化

最后给出总体总结。

"""


def _format_data_for_prompt(df: pd.DataFrame, technical_indicators: pd.DataFrame) -> str:
    """
    将数据格式化为提示词，对早期数据进行采样处理

    Args:
        df (pd.DataFrame): 原始股票数据
        technical_indicators (pd.DataFrame): 技术指标数据

    Returns:
        str: 格式化后的数据字符串
    """
    # 复制数据框以避免修改原始数据
    df_dict = df.copy()
    technical_indicators_dict = technical_indicators.copy()

    # 将时间索引转换为字符串格式
    df_dict.index = df_dict.index.strftime('%Y-%m-%d')
    technical_indicators_dict.index = technical_indicators_dict.index.strftime('%Y-%m-%d')

    # 分割数据：最近60天的数据和之前的数据
    recent_dates = list(df_dict.index)[-60:]
    early_dates = list(df_dict.index)[:-60]

    # 对早期数据进行采样（每2天取一个点）
    sampled_early_dates = early_dates[::2]

    # 合并采样后的日期和最近日期
    selected_dates = sampled_early_dates + recent_dates

    # 构建完整的数据字典，只包含选定的日期
    data_dict = {
        "历史数据": {
            date: {
                "开盘价": f"{df_dict.loc[date, 'open']:.2f}",
                "收盘价": f"{df_dict.loc[date, 'close']:.2f}",
                "最高价": f"{df_dict.loc[date, 'high']:.2f}",
                "最低价": f"{df_dict.loc[date, 'low']:.2f}",
                "成交量": f"{int(df_dict.loc[date, 'volume']):,}"
            } for date in selected_dates
        },
        "技术指标": {
            date: {
                "趋势指标": {
                    "MACD": f"{technical_indicators_dict.loc[date, 'MACD']:.2f}",
                    "DIF": f"{technical_indicators_dict.loc[date, 'DIF']:.2f}",
                    "DEA": f"{technical_indicators_dict.loc[date, 'DEA']:.2f}",
                    "MA5": f"{technical_indicators_dict.loc[date, 'MA5']:.2f}",
                    "MA10": f"{technical_indicators_dict.loc[date, 'MA10']:.2f}",
                    "MA20": f"{technical_indicators_dict.loc[date, 'MA20']:.2f}",
                    "MA60": f"{technical_indicators_dict.loc[date, 'MA60']:.2f}",
                    "TRIX": f"{technical_indicators_dict.loc[date, 'TRIX']:.2f}",
                    "TRMA": f"{technical_indicators_dict.loc[date, 'TRMA']:.2f}"
                },
                "摆动指标": {
                    "KDJ-K": f"{technical_indicators_dict.loc[date, 'K']:.2f}",
                    "KDJ-D": f"{technical_indicators_dict.loc[date, 'D']:.2f}",
                    "KDJ-J": f"{technical_indicators_dict.loc[date, 'J']:.2f}",
                    "RSI": f"{technical_indicators_dict.loc[date, 'RSI']:.2f}",
                    "CCI": f"{technical_indicators_dict.loc[date, 'CCI']:.2f}",
                    "BIAS1": f"{technical_indicators_dict.loc[date, 'BIAS1']:.2f}",
                    "BIAS2": f"{technical_indicators_dict.loc[date, 'BIAS2']:.2f}",
                    "BIAS3": f"{technical_indicators_dict.loc[date, 'BIAS3']:.2f}"
                },
                "布林带": {
                    "上轨": f"{technical_indicators_dict.loc[date, 'BOLL_UP']:.2f}",
                    "中轨": f"{technical_indicators_dict.loc[date, 'BOLL_MID']:.2f}",
                    "下轨": f"{technical_indicators_dict.loc[date, 'BOLL_LOW']:.2f}"
                },
                "动向指标": {
                    "PDI": f"{technical_indicators_dict.loc[date, 'PDI']:.2f}",
                    "MDI": f"{technical_indicators_dict.loc[date, 'MDI']:.2f}",
                    "ADX": f"{technical_indicators_dict.loc[date, 'ADX']:.2f}",
                    "ADXR": f"{technical_indicators_dict.loc[date, 'ADXR']:.2f}"
                },
                "成交量指标": {
                    "VR": f"{technical_indicators_dict.loc[date, 'VR']:.2f}",
                    "AR": f"{technical_indicators_dict.loc[date, 'AR']:.2f}",
                    "BR": f"{technical_indicators_dict.loc[date, 'BR']:.2f}",
                },
                "动量指标": {
                    "ROC": f"{technical_indicators_dict.loc[date, 'ROC']:.2f}",
                    "MAROC": f"{technical_indicators_dict.loc[date, 'MAROC']:.2f}",
                    "MTM": f"{technical_indicators_dict.loc[date, 'MTM']:.2f}",
                    "MTMMA": f"{technical_indicators_dict.loc[date, 'MTMMA']:.2f}",
                    "DPO": f"{technical_indicators_dict.loc[date, 'DPO']:.2f}",
                    "MADPO": f"{technical_indicators_dict.loc[date, 'MADPO']:.2f}"
                },
                "其他指标": {
                    "EMV": f"{technical_indicators_dict.loc[date, 'EMV']:.2f}",
                    "MAEMV": f"{technical_indicators_dict.loc[date, 'MAEMV']:.2f}",
                    "DIF_DMA": f"{technical_indicators_dict.loc[date, 'DIF_DMA']:.2f}",
                    "DIFMA_DMA": f"{technical_indicators_dict.loc[date, 'DIFMA_DMA']:.2f}"
                }
            } for date in selected_dates
        }
    }

    # 计算关键变化率
    latest_close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    last_week_close = df['close'].iloc[-6] if len(df) > 5 else prev_close
    last_month_close = df['close'].iloc[-21] if len(df) > 20 else prev_close

    data_dict["市场趋势"] = {
        "日涨跌幅": f"{((latest_close - prev_close) / prev_close * 100):.2f}%",
        "周涨跌幅": f"{((latest_close - last_week_close) / last_week_close * 100):.2f}%",
        "月涨跌幅": f"{((latest_close - last_month_close) / last_month_close * 100):.2f}%",
        "最新收盘价": f"{latest_close:.2f}",
        "最高价": f"{df['high'].max():.2f}",
        "最低价": f"{df['low'].min():.2f}",
        "平均成交量": f"{int(df['volume'].mean()):,}"
    }

    return json.dumps(data_dict, ensure_ascii=False, indent=2)


def _parse_analysis_response(analysis_text: str) -> Dict[str, Any]:
    """解析API返回的文本分析结果为结构化数据"""

    def clean_markdown(text: str) -> str:
        """清理格式并处理换行"""
        lines = text.split('\n')
        cleaned_lines = []

        for _line in lines:
            _line = _line.strip()
            if not _line:
                continue

            # 识别大标题
            if _line in ['技术分析', '走势分析', '投资建议', '风险提示', '总结', '总体总结']:
                continue

            # 处理数字标题
            if _line.startswith(('1.', '2.', '3.')):
                if cleaned_lines:
                    cleaned_lines.append('')  # 添加空行
                cleaned_lines.append(f'<p class="section-title">{_line}</p>')
                continue

            # 处理正文内容
            if ':' in _line:
                title, content = _line.split(':', 1)
                if content.strip():
                    cleaned_lines.append(f'<p class="item-title">{title}:</p>')
                    cleaned_lines.append(f'<p class="item-content">{content.strip()}</p>')
            else:
                cleaned_lines.append(f'<p>{_line}</p>')

        return '\n'.join(cleaned_lines)

    sections = {
        "技术分析": "",
        "走势分析": "",
        "投资建议": "",
        "风险提示": "",
        "总结": ""
    }

    current_section = None
    buffer = []

    # 按行处理文本
    for line in analysis_text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # 处理总结部分
        if line.startswith('总体总结'):
            if current_section:
                sections[current_section] = clean_markdown('\n'.join(buffer))
            current_section = "总结"
            buffer = [line.split('：', 1)[1] if '：' in line else line]
            continue

        # 处理主要部分
        if line in sections:
            if current_section and buffer:
                sections[current_section] = clean_markdown('\n'.join(buffer))
            current_section = line
            buffer = []
            continue

        if current_section:
            buffer.append(line)

    # 处理最后一个部分
    if current_section and buffer:
        sections[current_section] = clean_markdown('\n'.join(buffer))

    return {
        "AI分析结果": sections
    }


class APIBusyError(Exception):
    """API服务器繁忙时抛出的异常"""
    pass


class LLMAnalyzer:
    """使用 OpenAI SDK 与 llm API 交互的类"""

    def __init__(self, api_key: str, base_url: str, model: str = None):
        """
        初始化 llm 分析器

        Args:
            api_key (str): llm API 密钥
            base_url (str): llm API 基础 URL
            model (str): 使用的模型名称，默认为None则使用环境变量或空字符串
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client()
        )
        self.model = model or os.environ.get('LLM_MODEL', '')

    def request_analysis(self, df: pd.DataFrame, technical_indicators: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        向 llm API 发送分析请求

        Args:
            df (pd.DataFrame): 原始股票数据
            technical_indicators (pd.DataFrame): 技术指标数据

        Returns:
            Optional[Dict[str, Any]]: API 响应的分析结果
        """
        try:
            # 准备数据
            print("开始准备数据...")
            data_str = _format_data_for_prompt(df, technical_indicators)
            print(f"数据准备完成，数据长度: {len(data_str)}")

            # 构建消息
            print("构建API请求消息...")
            messages = [
                {"role": "system", "content": _create_system_prompt()},
                {"role": "user", "content": f"请分析以下股票数据并给出专业的分析意见：\n{data_str}"}
            ]
            print(f"消息构建完成，系统提示词长度: {len(messages[0]['content'])}")
            print(f"用户消息长度: {len(messages[1]['content'])}")

            # 发送请求
            print("开始发送API请求...")
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=1.0,
                    stream=False
                )
                print("API请求发送成功")
            except Exception as api_e:
                # 检查是否是空响应导致的JSON解析错误
                if str(api_e).startswith("Expecting value: line 1 column 1 (char 0)"):
                    print("API返回空响应，服务器可能繁忙")
                    raise APIBusyError("API服务器繁忙，返回空响应") from api_e
                print(f"API请求发送失败: {str(api_e)}")
                raise  # 重新抛出其他类型的异常

            # 记录原始响应以便调试
            print("API 原始响应类型:", type(response))
            print("API 原始响应内容:", response)

            # 检查响应内容
            if not response:
                print("API返回空响应")
                return format_analysis_result({})

            if not hasattr(response, 'choices'):
                print(f"API响应缺少choices属性，响应结构: {dir(response)}")
                return format_analysis_result({})

            if not response.choices:
                print("API响应的choices为空")
                return format_analysis_result({})

            # 解析响应
            try:
                analysis_text = response.choices[0].message.content
                print("成功获取分析文本内容")
                print("分析文本:", analysis_text)
            except Exception as text_e:
                print(f"获取分析文本失败: {str(text_e)}")
                raise

            # 将文本响应组织成结构化数据
            print("开始解析分析文本...")
            result = _parse_analysis_response(analysis_text)
            print("分析文本解析完成")
            return result

        except APIBusyError as be:  # 处理API繁忙异常
            print(f"=== API繁忙错误 ===")
            print(f"错误详情: {str(be)}")
            print(f"错误类型: {type(be)}")
            return format_analysis_result({})
        except json.JSONDecodeError as je:
            print(f"=== JSON解析错误 ===")
            print(f"错误详情: {str(je)}")
            print(f"错误类型: {type(je)}")
            print(f"错误位置: {je.pos}")
            print(f"错误行列: 行 {je.lineno}, 列 {je.colno}")
            print(f"错误的文档片段: {je.doc[:100] if je.doc else 'None'}")
            return format_analysis_result({})
        except openai.APITimeoutError as te:
            print(f"=== API超时错误 ===")
            print(f"错误详情: {str(te)}")
            print(f"错误类型: {type(te)}")
            return format_analysis_result({})
        except openai.APIConnectionError as ce:
            print(f"=== API连接错误 ===")
            print(f"错误详情: {str(ce)}")
            print(f"错误类型: {type(ce)}")
            return format_analysis_result({})
        except openai.APIError as ae:
            print(f"=== API错误 ===")
            print(f"错误详情: {str(ae)}")
            print(f"错误类型: {type(ae)}")
            return format_analysis_result({})
        except openai.RateLimitError as re:
            print(f"=== API频率限制错误 ===")
            print(f"错误详情: {str(re)}")
            print(f"错误类型: {type(re)}")
            return format_analysis_result({})
        except Exception as e:
            print(f"=== 未预期的错误 ===")
            print(f"错误详情: {str(e)}")
            print(f"错误类型: {type(e)}")
            print(f"错误追踪:")
            import traceback
            traceback.print_exc()
            return format_analysis_result({})
