"""Note data models - 笔记数据模型.

Pydantic models for Xiaohongshu note structure including:
- NoteMetaData: Content information (title, content, images, etc.)
- NotePrediction: Performance prediction metrics (CTR, sortScore, etc.)
- NoteTag: Platform-assigned tags (intention, taxonomy, marketing level)
- Note: Complete note combining all above
"""

from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl


class NoteMetaData(BaseModel):
    """笔记元数据 (Note metadata)."""

    note_id: str = Field(description="笔记ID")
    title: str = Field(description="笔记标题")
    content: str = Field(description="笔记正文内容")
    cover_image_url: str = Field(description="封面图片URL")
    inner_image_urls: Optional[List[str]] = Field(default=None, description="内页图片URLs (图文笔记)")
    video_url: Optional[str] = Field(default=None, description="视频URL (视频笔记)")
    nickname: Optional[str] = Field(default=None, description="作者昵称")

    # Platform statistics
    likes: Optional[int] = Field(default=None, description="点赞数")
    collects: Optional[int] = Field(default=None, description="收藏数")
    comments: Optional[int] = Field(default=None, description="评论数")
    shares: Optional[int] = Field(default=None, description="转发数")


class NotePrediction(BaseModel):
    """笔记预测指标 (Note prediction metrics).

    Performance predictions from content scoring models.
    """

    note_id: str = Field(description="笔记ID")

    # Primary metrics
    sort_score2: float = Field(description="平台优先级排序分数 (黑盒算法)", alias="sortScore")
    ctr: float = Field(description="点击率 (Click-through rate)")
    ces_rate: float = Field(description="五种互动行为率的加权和 (Composite engagement score)")

    # Engagement rates
    interaction_rate: float = Field(description="互动率")
    like_rate: float = Field(description="点赞率")
    fav_rate: float = Field(description="收藏率", alias="fav_rate")
    comment_rate: float = Field(description="评论率")
    share_rate: float = Field(description="转发率")
    follow_rate: float = Field(description="关注率")

    # Additional metrics
    impression: Optional[float] = Field(default=None, description="预估曝光量")

    class Config:
        populate_by_name = True  # Allow both 'sort_score2' and 'sortScore'


class NoteTag(BaseModel):
    """笔记标签 (Note tags).

    Platform-assigned tags from content classification models.
    """

    # Intention hierarchy
    intention_lv1: str = Field(description="一级意图 (e.g., '分享', '经验知识教程')")
    intention_lv2: str = Field(description="二级意图 (e.g., '推荐测评', '干货分享')")

    # Content taxonomy
    taxonomy1: str = Field(description="内容分类一级 (e.g., '母婴', '美妆')")
    taxonomy2: str = Field(description="内容分类二级 (e.g., '婴童食品', '婴童用品')")
    taxonomy3: str = Field(description="内容分类三级")

    # Marketing assessment
    note_marketing_integrated_level: str = Field(description="内容营销感 (e.g., '软广', '商品推荐')")


class Note(BaseModel):
    """完整笔记模型 (Complete note model).

    Combines meta_data, prediction, and tag for a complete note representation.
    """

    note_id: str = Field(description="笔记ID")
    meta_data: NoteMetaData = Field(description="笔记元数据")
    prediction: NotePrediction = Field(description="性能预测指标")
    tag: NoteTag = Field(description="平台标签")

    @classmethod
    def from_json(cls, data: dict) -> "Note":
        """从JSON数据创建Note实例 (Create Note instance from JSON data).

        Args:
            data: JSON dict with note_id, title, content, prediction, tag, etc.

        Returns:
            Note instance

        Example:
            >>> import json
            >>> with open('docs/owned_note.json') as f:
            ...     note_data = json.load(f)
            >>> note = Note.from_json(note_data)
        """
        return cls(
            note_id=data["note_id"],
            meta_data=NoteMetaData(
                note_id=data["note_id"],
                title=data["title"],
                content=data["content"],
                cover_image_url=data.get("cover_image_url", ""),
                inner_image_urls=data.get("inner_image_urls"),
                video_url=data.get("video_url"),
                nickname=data.get("nickname"),
                likes=data.get("likes"),
                collects=data.get("collects"),
                comments=data.get("comments"),
                shares=data.get("shares"),
            ),
            prediction=NotePrediction(**data["prediction"]),
            tag=NoteTag(**data["tag"]),
        )
