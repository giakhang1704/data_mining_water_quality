## Cách chạy
1. Cài đặt môi trường Python.
2. Cài dependencies:
```bash
pip install -r requirements.txt
```
3. Chạy dashboard:
```bash
streamlit run streamlit_app.py
```

## Nội dung
- `streamlit_app.py`: ứng dụng dashboard tương tác.
- `dataset/`: chứa dữ liệu nguồn (`weekly_land.csv`, `monthly_ocean.csv`, `metadata.xlsx`).
- `README.md`: hướng dẫn chạy.

## Gợi ý push lên GitHub
```bash
git init
git add .
git commit -m "Add water quality dashboard"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```
