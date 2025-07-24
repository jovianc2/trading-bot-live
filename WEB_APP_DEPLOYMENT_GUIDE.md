# Self-Learning Trading Bot Web Application Deployment Guide

This guide provides instructions for deploying the self-learning trading bot web application. The application consists of a React frontend and a Flask backend.

## Prerequisites

-   Python 3.8+
-   Node.js 16+
-   npm or pnpm (for frontend dependencies)
-   pip (for Python dependencies)
-   Git (optional, for version control)
-   Docker and Docker Compose (recommended for production deployment)

## Project Structure

```
./trading_bot_web_app_package/
├── trading-bot-frontend/  # React frontend application
│   ├── public/
│   ├── src/
│   ├── dist/              # Built frontend files (after `pnpm run build`)
│   ├── ...
├── trading-bot-backend/   # Flask backend application
│   ├── venv/              # Python virtual environment
│   ├── src/
│   │   ├── static/        # Frontend static files (copied from trading-bot-frontend/dist)
│   │   ├── main.py        # Flask application entry point
│   │   ├── ...            # Other Python modules (ai_model.py, risk_management.py, etc.)
│   ├── requirements.txt   # Python dependencies
│   ├── ...
└── WEB_APP_DEPLOYMENT_GUIDE.md  # This guide
```

## Deployment Steps

### Option 1: Manual Deployment (Recommended for development/testing)

#### 1. Backend Setup

1.  Navigate to the backend directory:
    ```bash
    cd trading_bot_web_app_package/trading-bot-backend
    ```
2.  Create and activate a Python virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Ensure the core trading bot Python modules are present in the `trading-bot-backend/` directory. They should have been copied there during the build process.

5.  Start the Flask backend server:
    ```bash
    python src/main.py
    ```
    The backend will run on `http://0.0.0.0:5000`.

#### 2. Frontend Setup (if not already built and copied)

If you need to rebuild the frontend or if the `dist` folder is missing in `trading-bot-frontend`:

1.  Navigate to the frontend directory:
    ```bash
    cd trading_bot_web_app_package/trading-bot-frontend
    ```
2.  Install Node.js dependencies:
    ```bash
    pnpm install
    ```
3.  Build the React frontend for production:
    ```bash
    pnpm run build
    ```
    This will create a `dist` directory with the optimized static files.

4.  Copy the built frontend files to the Flask static directory:
    ```bash
    cp -r dist/* ../trading-bot-backend/src/static/
    ```

#### 3. Access the Web Application

Once the Flask backend is running and serving the static frontend files, you can access the web application by opening your web browser and navigating to the IP address or domain where your Flask application is hosted, on port `5000` (e.g., `http://your_server_ip:5000`).

### Option 2: Docker Deployment (Recommended for production)

For a more robust and scalable production deployment, it is recommended to containerize the application using Docker and Docker Compose.

#### 1. Create Dockerfiles

**`trading-bot-backend/Dockerfile`**

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt .
COPY src/ ./src/
COPY main_controller.py .
COPY ai_model.py .
COPY risk_management.py .
COPY automated_retraining.py .
COPY python_server.py .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["python", "src/main.py"]
```

**`trading-bot-frontend/Dockerfile`**

```dockerfile
# Use an official Node.js runtime as a parent image
FROM node:20-alpine

# Set the working directory in the container
WORKDIR /app

# Copy package.json and pnpm-lock.yaml to install dependencies
COPY package.json pnpm-lock.yaml ./

# Install pnpm
RUN npm install -g pnpm

# Install dependencies
RUN pnpm install --frozen-lockfile

# Copy the rest of the application code
COPY . .

# Build the React app
RUN pnpm run build

# Use a lightweight web server to serve the static files
FROM nginx:alpine
COPY --from=0 /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

#### 2. Create `docker-compose.yml`

In the root of `trading_bot_web_app_package` (or a level above, depending on your preference), create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  backend:
    build:
      context: ./trading-bot-backend
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    volumes:
      - ./trading-bot-backend:/app # Mount for development, remove for production
    environment:
      # Add any environment variables needed for your Flask app
      PYTHONUNBUFFERED: 1

  frontend:
    build:
      context: ./trading-bot-frontend
      dockerfile: Dockerfile
    ports:
      - "80:80"
    depends_on:
      - backend
```

#### 3. Build and Run with Docker Compose

1.  Navigate to the `trading_bot_web_app_package` directory:
    ```bash
    cd trading_bot_web_app_package
    ```
2.  Build and run the Docker containers:
    ```bash
    docker-compose up --build -d
    ```
    The `-d` flag runs the containers in detached mode.

3.  Access the web application by opening your web browser and navigating to your server's IP address or domain (port 80 for the frontend, e.g., `http://your_server_ip`).

#### 4. Stop and Remove Containers

To stop the running containers:

```bash
docker-compose down
```

## Important Notes

-   **Security**: For production environments, ensure you implement proper HTTPS, secure your API keys, and follow best security practices.
-   **Data Persistence**: If your trading bot needs to store persistent data (e.g., trade logs, model checkpoints), ensure you configure Docker volumes or bind mounts appropriately.
-   **NinjaTrader Integration**: The current setup assumes NinjaTrader runs on a separate machine or within a network accessible to the backend. Ensure network connectivity and firewall rules allow communication between the Flask backend and your NinjaTrader instance (via the `python_server.py` component).
-   **Real-time Data**: The WebSocket connection between the frontend and backend provides real-time updates. Ensure your server infrastructure supports WebSocket connections.

If you encounter any issues during deployment, please refer to the logs of the respective services (`docker-compose logs backend` or `docker-compose logs frontend`).


