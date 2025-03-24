SWAGGER_TEMPLATE = {
    "swagger": "2.0",
    "info": {
        "title": "단무지 AI API",
        "description": "단지, 단무지와 함께해요",
        "version": "1.0.0",
    },
    "tags": [
        {
            "name": "Face API",
            "description": "얼굴 이미지 등록 및 인물 식별 관련 기능을 제공합니다.",
        }
    ],
    "securityDefinitions": {
        "Bearer": {
            "type": "apiKey",
            "name": "Authorization",
            "in": "header",
            "description": "JWT 토큰 형식: Bearer <token>",
        }
    },
    "security": [{"Bearer": []}],
}

SWAGGER_CONFIG = {
    "uiversion": 3,
    "title": "단무지 AI API",
    "layout": "BaseLayout"
}
