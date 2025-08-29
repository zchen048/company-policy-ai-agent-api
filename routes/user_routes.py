from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from ..database import get_session
from ..schemas.user_schemas import CreateUser, ReadUser, UpdateUser
from ..logic.user_logic import create_user, get_user_by_id
from ..models import RankEnum

router = APIRouter()

@router.post("/users/", response_model=ReadUser)
def create_user_endpoint(user: CreateUser, session: Session = Depends(get_session)):
    return create_user(
        session=session, 
        name=user.name, 
        email=user.email,
        department=user.department,
        rank=user.rank,
        title=user.title
    )

@router.get("/users/{user_id}", response_model=ReadUser)
def get_user_by_id_endpoint(user_id: int, session: Session = Depends(get_session)):
    user = get_user_by_id(session=session, id=user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.get("/users/", response_model=List[ReadUser])
def get_users_endpoint(
    session: Session = Depends(get_session),
    name: Optional[str] = None,
    email: Optional[str] = None,
    department: Optional[str] = None,
    rank: Optional[RankEnum] = None,
    title: Optional[str] = None
):
    return get_users(
        session=session, 
        name=name, 
        email=email,
        department=department,
        rank=rank,
        title=title
    )

@router.patch("/users/", response_model=ReadUser)
def update_user_endpoint(id:int, to_change:UpdateUser, session: Session = Depends(get_session)):
    try:
        user = update_user(session=session, id=id, data=to_change)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

@router.delete("/users/{user_id}", status_code=204)
def delete_user_endpoint(user_id: int, session: Session = Depends(get_session)):
    deleted = delete_user(session=session, id=user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")