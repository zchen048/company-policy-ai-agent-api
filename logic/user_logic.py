from typing import Optional, List, Union
from sqlmodel import Session, select
from ..models import RankEnum, User
from ..schemas.user_schemas import UpdateUser
from ..utils import sg_datetime
from ..logger import get_logger
from ..exceptions import UserNotFoundException, NoFieldsToUpdateException

logger = get_logger(__name__)

def create_user(
    session: Session, 
    name: str, 
    email: str,
    department:str,
    rank: RankEnum,
    title: str
) -> User:
    """ Create and persist a new user in the database. Id will be generated. """

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
    logger.info(f"User `{name}` created ")
    return user

def get_user_by_id(session: Session, id: int):
    """ Get user with id from database. """
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
    """ Get all users with certain field from database. """
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
    """ Update specified fields of user with a specific id. """
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
    logger.info(f"User `{name}` updated ")
    return user

def delete_user(session: Session, id: int):
    """ Remove user with a specific id. """
    user = session.get(User, id)
    if not user:
       raise UserNotFoundException()
    
    session.delete(user)
    session.commit()
    logger.info(f"User with `{id}` deleted ")
