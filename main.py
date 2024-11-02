from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sqlalchemy import Column, Float, Integer, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

DATABASE_URL = "sqlite:///./water_levels.db"

# Set up the database and SQLAlchemy
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()

# Create a table model for water levels
class WaterLevelModel(Base):
    __tablename__ = "water_levels"
    id = Column(Integer, primary_key=True, index=True)
    level = Column(Float, nullable=False)

# Create the tables
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI()

# CORS settings
origins = [
    "http://localhost",  # Allow requests from localhost
    "http://localhost:3000",  # Example for allowing a frontend on a different port
    "https://yourfrontenddomain.com"  # Replace with your frontend domain
]

# Middleware to handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Pydantic model for single and multiple water levels
class WaterLevel(BaseModel):
    level: float

class WaterLevels(BaseModel):
    levels: List[WaterLevel]

# Dependency to get the database session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to load historical water levels from the database
def load_water_levels(db: Session):
    water_levels = db.query(WaterLevelModel.level).all()
    return [level[0] for level in water_levels]

# Forecast function
def forecast_water_levels(data, days_ahead=15):
    data_series = pd.Series(data)
    model = ARIMA(data_series, order=(5, 1, 0))  # Adjust p, d, q as needed
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=days_ahead)
    return forecast.tolist()

# Endpoint to add a single water level entry
@app.post("/water-level/")
def add_water_level(data: WaterLevel, db: Session = Depends(get_db)):
    new_level = WaterLevelModel(level=data.level)
    db.add(new_level)
    db.commit()
    db.refresh(new_level)
    return {"message": "Water level added successfully", "level": new_level.level}

# Endpoint to add multiple water level entries
@app.post("/water-levels/")
def add_multiple_water_levels(data: WaterLevels, db: Session = Depends(get_db)):
    new_levels = [WaterLevelModel(level=entry.level) for entry in data.levels]
    db.add_all(new_levels)
    db.commit()
    return {"message": "Water levels added successfully", "levels": [entry.level for entry in new_levels]}

# Endpoint to retrieve the latest water level entry
@app.get("/water-level/latest/")
def get_latest_water_level(db: Session = Depends(get_db)):
    result = db.query(WaterLevelModel).order_by(WaterLevelModel.id.desc()).first()
    if not result:
        raise HTTPException(status_code=404, detail="No water level data available")
    return {"latest_level": result.level}

# Endpoint to retrieve all water level entries
@app.get("/water-level/all/")
def get_all_water_levels(db: Session = Depends(get_db)):
    results = db.query(WaterLevelModel).all()
    if not results:
        raise HTTPException(status_code=404, detail="No water level data available")
    return {"all_levels": [entry.level for entry in results]}

# Endpoint for forecasting the next 15 days of water levels
@app.get("/water-level/forecast/")
def forecast_next_15_days(db: Session = Depends(get_db)):
    historical_data = load_water_levels(db)
    if len(historical_data) < 10:
        return {"error": "Not enough data to make a prediction. Please add more data points."}
    
    prediction = forecast_water_levels(historical_data, days_ahead=15)
    forecast = {f"day_{i+1}": level for i, level in enumerate(prediction)}
    return {"15_day_forecast": forecast}
