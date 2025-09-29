from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from ..database import get_session
from ..schemas.user_schemas import CreateUser, ReadUser, UpdateUser
from ..logic.user_logic import create_user, get_user_by_id
from ..models import RankEnum
from ..exceptions import UserNotFoundException, NoFieldsToUpdateException

router = APIRouter()

@router.post("/users/", response_model=ReadUser)
def create_user_endpoint(user: CreateUser, session: Session = Depends(get_session)):
    """ API endpoint to create a new user and return the created user's info. """
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
    """ API endpoint to get a user info using id. """
    try:
        return get_user_by_id(session=session, id=user_id)
    except UserNotFoundException:
        raise HTTPException(status_code=404, detail="User not found")

@router.get("/users/", response_model=List[ReadUser])
def get_users_endpoint(
    session: Session = Depends(get_session),
    name: Optional[str] = None,
    email: Optional[str] = None,
    department: Optional[str] = None,
    rank: Optional[RankEnum] = None,
    title: Optional[str] = None
):
    """ API endpoint to get all users with certain fields. """
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
    """ API endpoint update info of user of a particular id. """
    try:
        user = update_user(session=session, id=id, data=to_change)
    except UserNotFoundException:
        raise HTTPException(status_code=404, detail="User not found")
    except NoFieldsToUpdateException:
        raise HTTPException(status_code=400, detail="No fields provided to update")    
    return user

@router.delete("/users/{user_id}", status_code=204)
def delete_user_endpoint(user_id: int, session: Session = Depends(get_session)):
    """ API endpoint to delete user of a particular id. """
    try:
        delete_user(session=session, id=user_id)
    except UserNotFoundException:
        raise HTTPException(status_code=404, detail="User not found")