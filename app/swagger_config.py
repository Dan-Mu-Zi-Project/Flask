SWAGGER_TEMPLATE = {
    "swagger": "2.0",
    "info": {
        "title": "단무지 AI API",
        "description": "단지, 단무지와 함께해요",
        "version": "1.0.0",
    },
    # "host": "api.danmujii.com",
    # "schemes": ["http"],
    # "basePath": "/api",
    "tags": [
        {
            "name": "Face API",
            "description": "얼굴 이미지 등록 및 인물 식별 관련 기능을 제공합니다.",
        },
        {
            "name": "Photo Feedback API",
            "description": "GPT 기반 단체사진 분석 및 음성 피드백 기능을 제공합니다.",
        },
        {
            "name": "Photo Upload API",
            "description": "사진 업로드, Presigned URL 처리, 인물 자동 태깅 기능을 제공합니다.",
        },
        {
            "name": "Test API",
            "description": "테스트 기능으로 배포 서버에서 사용은 하지 않습니다.",
        },
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
    "layout": "BaseLayout",
    "docExpansion": "list",
    "displayRequestDuration": True,
}
