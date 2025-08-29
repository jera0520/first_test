#!/usr/bin/env python3
"""
SKINMATE 애플리케이션 실행 스크립트
"""

from skinmate_app import create_app

if __name__ == '__main__':
    app = create_app()
    
    # 업로드 디렉토리 생성
    import os
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    app.run(debug=True, port=5001)
