"""
直接测试工具调用（不使用pytest框架）
使用 docs/target_notes.json 中的真实数据测试 NLPAnalysisTool 和 MultiModalVisionTool
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 加载环境变量
load_dotenv()

# 导入模型和工具
from src.xhs_seo_optimizer.models.note import Note
from src.xhs_seo_optimizer.tools.nlp_analysis import NLPAnalysisTool
from src.xhs_seo_optimizer.tools.multimodal_vision import MultiModalVisionTool


def load_notes_from_json(json_path: str, limit: int = None):
    """从JSON文件加载笔记数据"""
    with open(json_path, "r", encoding="utf-8") as f:
        notes_data = json.load(f)

    notes = [Note.from_json(note_data) for note_data in notes_data]

    if limit:
        notes = notes[:limit]

    return notes


def test_nlp_tool(notes):
    """测试NLP分析工具"""
    print("\n" + "="*80)
    print("开始测试 NLPAnalysisTool")
    print("="*80)

    # 初始化工具
    nlp_tool = NLPAnalysisTool()

    for i, note in enumerate(notes, 1):
        print(f"\n{'='*80}")
        print(f"笔记 {i}/{len(notes)}: {note.note_id}")
        print(f"标题: {note.meta_data.title}")
        print(f"内容长度: {len(note.meta_data.content)} 字符")
        print("="*80)

        try:
            # 调用工具的 _run 方法
            result_json = nlp_tool._run(
                note_meta_data=note.meta_data,
                note_id=note.note_id
            )

            # 解析JSON结果
            result = json.loads(result_json)

            print("\n✅ NLP 分析成功！")
            print(f"\n📊 分析结果摘要:")
            print(f"  - 标题模式: {result.get('title_pattern', 'N/A')}")
            print(f"  - 标题关键词: {', '.join(result.get('title_keywords', []))}")
            print(f"  - 开头策略: {result.get('opening_strategy', 'N/A')}")
            print(f"  - 内容框架: {result.get('content_framework', 'N/A')}")
            print(f"  - 结尾技巧: {result.get('ending_technique', 'N/A')}")
            print(f"  - 字数统计: {result.get('word_count', 0)}")
            print(f"  - 痛点挖掘: {', '.join(result.get('pain_points', []))}")
            print(f"  - 情感触发: {', '.join(result.get('emotional_triggers', []))}")

            # 保存完整结果
            output_dir = Path(__file__).parent / "output"
            output_dir.mkdir(exist_ok=True)

            output_file = output_dir / f"nlp_result_{note.note_id}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"\n💾 完整结果已保存到: {output_file}")

        except Exception as e:
            print(f"\n❌ NLP 分析失败: {str(e)}")
            import traceback
            traceback.print_exc()


def test_vision_tool(notes):
    """测试视觉分析工具"""
    print("\n" + "="*80)
    print("开始测试 MultiModalVisionTool")
    print("="*80)

    # 初始化工具
    vision_tool = MultiModalVisionTool(max_inner_images=4)

    for i, note in enumerate(notes, 1):
        print(f"\n{'='*80}")
        print(f"笔记 {i}/{len(notes)}: {note.note_id}")
        print(f"标题: {note.meta_data.title}")

        # 统计图片数量
        inner_count = len(note.meta_data.inner_image_urls) if note.meta_data.inner_image_urls else 0
        total_images = 1 + min(4, inner_count)  # 1封面 + 最多4张内页

        print(f"图片数量: 封面图 1 张 + 内页图 {inner_count} 张")
        print(f"分析范围: {total_images} 张图片")
        print("="*80)

        try:
            # 调用工具的 _run 方法
            result_json = vision_tool._run(
                note_meta_data=note.meta_data,
                note_id=note.note_id
            )

            # 解析JSON结果
            result = json.loads(result_json)

            print("\n✅ 视觉分析成功！")
            print(f"\n📊 分析结果摘要:")
            print(f"  - 图片数量: {result.get('image_count', 0)}")
            print(f"  - 图片质量: {result.get('image_quality', 'N/A')}")
            print(f"  - 图片风格: {result.get('image_style', 'N/A')}")
            print(f"  - 色彩方案: {result.get('color_scheme', 'N/A')}")
            print(f"  - 排版风格: {result.get('layout_style', 'N/A')}")
            print(f"  - 封面吸引力: {result.get('thumbnail_appeal', 'N/A')}")
            print(f"  - 滑动停留力: {result.get('scroll_stopping_power', 'N/A')}")
            print(f"  - OCR文字: {result.get('text_ocr_content', 'N/A')[:100]}...")

            # 保存完整结果
            output_dir = Path(__file__).parent / "output"
            output_dir.mkdir(exist_ok=True)

            output_file = output_dir / f"vision_result_{note.note_id}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"\n💾 完整结果已保存到: {output_file}")

        except Exception as e:
            print(f"\n❌ 视觉分析失败: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    # 检查API密钥
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ 错误: 未找到 OPENROUTER_API_KEY 环境变量")
        print("请在 .env 文件中设置 OPENROUTER_API_KEY")
        return

    print("✅ 找到 OPENROUTER_API_KEY")

    # 加载笔记数据
    json_path = Path(__file__).parent.parent / "docs" / "target_notes.json"

    print(f"\n📖 从 {json_path} 加载笔记数据...")

    # 测试3个笔记
    notes = load_notes_from_json(str(json_path), limit=3)

    print(f"✅ 成功加载 {len(notes)} 个笔记")

    # 直接测试两个工具（不需要交互式输入）
    print("\n" + "="*80)
    print("开始测试两个工具: NLPAnalysisTool + MultiModalVisionTool")
    print("="*80)

    test_nlp_tool(notes)
    #test_vision_tool(notes)

    print("\n" + "="*80)
    print("✅ 测试完成！")
    print("="*80)


if __name__ == "__main__":
    main()
