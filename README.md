## CI/CD

This project uses GitHub Actions for continuous integration.

Every push to the `main` branch automatically:

1. Installs Python dependencies
2. Checks Python syntax
3. Runs the FastAPI API tests

The application is deployed using Render:

- FastAPI backend
- Streamlit frontend