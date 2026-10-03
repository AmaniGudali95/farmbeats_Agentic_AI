import os
from datetime import datetime,timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

SECRET_KEY = os.environ.get("JWT_SECRET", "farmbeats-dev-secret-change-in-production")
ALGORITHM="HS256"
TOKEN_EXPIRE_HOURS=24

pwd_context=CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme=OAuth2PasswordBearer(tokenUrl="login")

USERS={}

class UserRegister(BaseModel):
    name: str
    email: str
    password: str
    fields: list[str]=["field_a"]

class UserLogin(BaseModel):
    email: str
    password: str
class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    name: str
    fields: list
def hash_password(password):
    return pwd_context.hash(password)

def verify_password(plain, hashed):
    return pwd_context.verify(plain,hashed)

def create_token(user_id):
    expires=datetime.utcnow()+timedelta(hours=TOKEN_EXPIRE_HOURS)
    payload={"sub": user_id, "exp": expires}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token=Depends(oauth2_scheme)):
    credentials_error= HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload=jwt.decode(token,SECRET_KEY,algorithms=[ALGORITHM])
        email=payload.get("sub")
        if email is None or email not in USERS:
            raise credentials_error
        return USERS[email]
    except JWTError:
        raise credentials_error
