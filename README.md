# Financial Agent

A Python-based financial data processing and analysis tool that uses AI agents to clean, process, and search through financial data files.

## Features

- **Data Cleaning Agent**: Automatically processes and cleans Excel files from the `data/` directory
- **Simple Search Agent**: Searches and analyzes cleaned CSV data files
- **Web Interface**: Interactive playground interface for agent interactions
- **PostgreSQL with pgvector**: Vector database for enhanced data storage and retrieval

## Prerequisites

- Python 3.12 or higher
- Docker and Docker Compose (for PostgreSQL database)
- UV package manager (recommended) or pip

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd financial_agent
   ```

2. **Set up Python environment**

   **Option A: Using UV (recommended)**

   ```bash
   # Install UV if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Create virtual environment and install dependencies
   uv sync
   ```

   **Option B: Using pip**

   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install dependencies
   pip install -e .
   ```

3. **Start the PostgreSQL database**

   ```bash
   docker-compose up -d
   ```

   This will start a PostgreSQL instance with pgvector extension on port 5532.

## Project Structure

```
financial_agent/
├── data/                    # Input Excel files
│   ├── Estado_cuenta.xlsx
│   ├── facturas.xlsx
│   └── gastos_fijos.xlsx
├── src/
│   ├── agents/             # AI agents
│   │   ├── data_cleaner_agent.py
│   │   └── simple_search_agent.py
│   ├── tools/              # Data processing tools
│   │   └── data_cleaning_tools.py
│   └── main.py             # Main application entry point
├── docker-compose.yml      # Database configuration
├── pyproject.toml          # Project dependencies
└── README.md
```

## Usage

### 1. Data Processing

The project includes two main agents:

- **Data Cleaner Agent**: Processes Excel files from the `data/` directory and creates cleaned CSV files in `data_cleaned/`
- **Simple Search Agent**: Searches and analyzes the cleaned CSV data

### 2. Running the Application

Start the web interface:

```bash
# Activate virtual environment (if using pip)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run the application
python src/main.py
```

The application will start a web server (typically on `http://localhost:8000`) where you can interact with the agents through a playground interface.

### 3. Using the Agents

1. **Data Cleaning**: Use the Data Cleaner Agent to process your Excel files

   - Place your Excel files in the `data/` directory
   - The agent will automatically detect and clean the files
   - Cleaned files will be saved in `data_cleaned/` directory

2. **Data Search**: Use the Simple Search Agent to query your cleaned data
   - The agent can search through all cleaned CSV files
   - Ask questions about your financial data
   - Get insights and analysis

## Configuration

### Database Configuration

The PostgreSQL database is configured with the following default settings:

- **Host**: localhost
- **Port**: 5532
- **Database**: ai
- **Username**: ai
- **Password**: ai

You can modify these settings in `docker-compose.yml` if needed.

### Environment Variables

If you need to configure additional settings, you can set environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key (if using OpenAI models)
- `DATABASE_URL`: Custom database connection string
