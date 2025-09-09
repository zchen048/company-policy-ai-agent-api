from typing import Optional, List, Union
from sqlmodel import Session, select
from ..models import RankEnum, User
from ..schemas.user_schemas import UpdateUser
from ..utils import sg_datetime
from ..exceptions import UserNotFoundException, NoFieldsToUpdateException

def create_user(
    session: Session, 
    name: str, 
    email: str,
    department:str,
    rank: RankEnum,
    title: str
) -> User:

    user = User(
        name=name, 
        email=email,
        department=department,
        rank=rank,
        title=title
    )
    session.add(user)
    session.commit()
    session.refresh(user)  # refresh to get generated ID
    return user

def get_user_by_id(session: Session, id: int):
    user = session.get(User, id)
    if not user:
        raise UserNotFoundException()
    return user

def get_users(
    session: Session,
    name: Optional[str],
    email: Optional[str],
    department: Optional[str],
    rank:Optional[RankEnum],
    title:Optional[str]
)-> Union[List[User], None]:
    
    statement = select(User)

    if name:
        statement = statement.where(User.name == name)
    if name:
        statement = statement.where(User.email == email)
    if department:
        statement = statement.where(User.department == department)
    if rank:
        statement = statement.where(User.rank == rank)
    if title:
        statement = statement.where(User.title == title)
    
    return session.exec(statement).all()

def update_user(
    session: Session,
    id: int,
    data: UpdateUser
) -> User:
    
    user = session.get(User, id)
    if not user:
        raise UserNotFoundException()
    
    updates = data.model_dump(exclude_unset=True)
    if not updates:
        raise NoFieldsToUpdateException()

    for field, value in updates.items():
        setattr(user, field, value)
    
    user.modified_at = sg_datetime.get_sgt_time()

    session.add(user)
    session.commit()
    session.refresh(user)
    return user

def delete_user(session: Session, id: int):
    user = session.get(User, id)
    if not user:
       raise UserNotFoundException()
    
    session.delete(user)
    session.commit()
