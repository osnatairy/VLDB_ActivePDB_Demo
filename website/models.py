from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func


class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.String(10000))
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    notes = db.relationship('Note')

class Variable(db.Model):
    transaction_id = db.Column(db.String(5), primary_key=True)
    Transaction_name = db.Column(db.String(10))
    id =db.Column(db.Integer)
    id_gen =db.Column(db.String(20))
    entity =db.Column(db.String(150))
    relation =db.Column(db.String(150))
    value =db.Column(db.String(150))
    probability =db.Column(db.Float)
    source =db.Column(db.String(250))
    action = db.Column(db.String(25))


class Boolean(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    bool_id = db.Column(db.Integer)
    dnf = db.Column(db.String(250))
    original_dnf = db.Column(db.String(250))
    query_id = db.Column(db.String(3), db.ForeignKey('query.id'))
    name_results = db.Column(db.String(500))

class Query(db.Model):
    id = db.Column(db.String(3), primary_key=True)  # name
    name = db.Column(db.String(3))
    description = db.Column(db.String(1000))
    body = db.Column(db.String(1500))
    name_results = db.Column(db.String(500))
    short_des = db.Column(db.String(200))
    booleans = db.relationship('Boolean')

booleanQuery = db.Table('BooleanQuery',
    db.Column('boolId', db.Integer, db.ForeignKey('boolean.id'), primary_key=True),
    db.Column('queryId', db.Integer, db.ForeignKey('query.id'), primary_key=True)
)
#
#
# class Ans(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#
