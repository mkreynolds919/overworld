from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request

from overworld.api.catalog import Catalog
from overworld.api.schemas import GameDetail, GameSummary, Health


def create_app(catalog: Catalog | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if catalog is None:
            app.state.catalog = Catalog.load()
        yield

    app = FastAPI(title="Overworld", lifespan=lifespan)
    if catalog is not None:
        app.state.catalog = catalog

    def get_catalog(request: Request) -> Catalog:
        return request.app.state.catalog

    def require_game(game_id: int, cat: Catalog) -> None:
        if not cat.has(game_id):
            raise HTTPException(status_code=404, detail=f"no game with id {game_id}")

    @app.get("/health", response_model=Health)
    def health(cat: Catalog = Depends(get_catalog)):
        return {"status": "ok", "games": len(cat.games)}

    @app.get("/games/search", response_model=list[GameSummary])
    def search(
        q: str = Query(min_length=1, max_length=100),
        limit: int = Query(10, ge=1, le=50),
        cat: Catalog = Depends(get_catalog)
    ):
        return [cat.summary(g) for g in cat.search(q, limit)]

    @app.get("/games/{game_id}", response_model=GameDetail)
    def game(game_id: int, cat: Catalog = Depends(get_catalog)):
        require_game(game_id, cat)
        return cat.detail(game_id)

    return app

app = create_app()