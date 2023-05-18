from sqlalchemy import Column, String, create_engine,Float,Integer
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import NoResultFound



# 创建对象的基类:
Base = declarative_base()

# 初始化数据库连接:
engine = create_engine('mysql+pymysql://redbook:heibulindalizi@47.113.198.0:3306/OpenCV')
# 创建DBSession类型:
DBSession = sessionmaker(bind=engine)

class User(Base):
    # 表的名字:
    __tablename__ = 'User'

    # 表的结构:
    user_id = Column(Integer,primary_key=True,autoincrement=True)
    user_name = Column(String(20))
    allowed = Column(Integer)
    image = Column(Integer)

    def __init__(self,user_id,user_name,allowed,image):
        self.user_id = user_id
        self.user_name = user_name
        self.allowed = allowed
        self.image = image

    @staticmethod
    def insertUser(user):
        session = DBSession()
        try:
            session.add(user)
            session.commit()
            return user.user_id
        except Exception:
            return 0
        finally:
            session.close()

    @staticmethod
    def getUser(userid : int):
        session = DBSession()
        try:
            user = session.query(User).filter(User.user_id == userid).one()
            return user
        except NoResultFound:
            return 0
        finally:
            session.close()

    @staticmethod
    def updateUser(user):
        session = DBSession()
        try:
            session.add(user)
            session.commit()
            return 1
        except Exception:
            return 0
        finally:
            session.close()