from pydantic import BaseModel


class GameSummary(BaseModel):
    id: int
    display_name: str
    year: int | None
    cover_url: str | None


class Version(BaseModel):
    id: int
    display_name: str


class GameDetail(GameSummary):
    name: str
    platforms: list[str]
    genres: list[str]
    themes: list[str]
    series: list[str]
    developers: list[str]
    versions: list[Version]
    summary: str | None


class Health(BaseModel):
    status: str
    games: int