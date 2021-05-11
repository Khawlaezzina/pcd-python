import csv

from flask import Flask, jsonify
import pymysql
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

conn=pymysql.connect(host='localhost',user='root',password='',database='pcd')
cursor=conn.cursor()
query1='select id,rating from freelancer'
query2='select * from mission where hired=false'
query3='select * from client'
query4='select name from skills'
query5='select * from skilled'

#defining a list containing all the name skills presented on the table skills
def generateSkillsList():
  cursor.execute(query4)
  rows=cursor.fetchall()
  skills=[]
  for row in rows:
    skills.append(row[0])
  return skills

#defining a matrix containing all non hired missions on each row we find [idMission,skills,ratingClient]
def generateNonHiredMissionMatrix():
  skills=generateSkillsList()
  cursor.execute(query2)
  missions=cursor.fetchall()
  VectorMissionHeader=["idMission"]
  MissionMatrix1=[]
  for mission in missions :
    MissionVector=[mission[0]]
    technologies=mission[6].split("/")  #I should verify the number of column technologies on table mission
    for skill in skills:
      VectorMissionHeader.append(skill)
      if skill in technologies :
          MissionVector.append(1)
      else :
        MissionVector.append(0)
    cursor.execute('select rating from client where id_client=(select id_client from mission where id_mission=%s)',mission[0]);
    result=cursor.fetchall()
    rating=result[0][0]
    MissionVector.append(rating)
    MissionMatrix1.append(MissionVector)
  MissionMatrix=np.array(MissionMatrix1)
  return MissionMatrix
#defining a freelancers Matrix containg on each row [idFreelancer,skills,rating,nbreprojets]
def generateFreelancersMatrix():
  skills = generateSkillsList()
  cursor.execute(query1)
  freelancers=cursor.fetchall()
  freelancersMatrix1=[]
  Freelancerheaders = ["idfreelancer"]
  for skill in skills :
    Freelancerheaders.append(skill)
  Freelancerheaders.append("rating")
  Freelancerheaders.append("NombreProjets")
  for freelancer in freelancers:
    query = 'select skills.name from skills where id_skill in (select id_skill from skilled where freelancer_id=%s',freelancer[0]
    cursor.execute(query)
    result1 = cursor.fetchall()
    freelancerSkills=[]
    for result in result1 :
      freelancerSkills.append(result[0])
    freelancerList = [freelancer[0]]
    for skill in skills:
      if (skill in freelancerSkills):
        freelancerList.append(1)
      else:
        freelancerList.append(0)

    freelancerList.append(freelancer[1])
    embeddedquery='select count(id_mission) from mission where completed=true and id_freelancer=%s',freelancer[0]
    cursor.execute(embeddedquery)
    result2=cursor.fetchall()
    nombreProjets=result2[0][0]

    freelancerList.append(nombreProjets)
  freelancersMatrix1.append(freelancerList)
  freelancersMatrix=np.array(freelancersMatrix1)
  return freelancersMatrix
freelancers= {
  "data": [
    {
      "name":"Khawla",
      "firstName":"Ezzina",
      "job":"web developper"
    },
    {
      "name": "Nirmine",
      "firstName": "Khaled",
      "job": "web developper"
    }
  ]
}

def calculRatingNombreProjetFactor(rating,pourcentageRating,nombreProjet,pourcentageNombreProjet):
  return (pourcentageRating/100)*rating+(pourcentageNombreProjet/100)*nombreProjet

@app.route('/',methods=['GET'])
def hello_world():
    return 'Hello World!'

@app.route('/freelancers/',methods=['GET'])
def freelancersReport():
  global freelancers
  return jsonify([freelancers])
#recommand offers to a particular freelancer
@app.route('/recommandOffer/<idfreelancer>')
def recommandOffers(idfreelancer):
  query='select skills.name from skills where id_skill in (select id_skill from skilled where freelancer_id=%s', idfreelancer
  cursor.execute(query)
  result1 = cursor.fetchall()
  freelancerSkills = []
  headers = ['idfreelancer']
  skills = generateSkillsList()
  for skill in skills :
    headers.append(skill)
  for result in result1:
    freelancerSkills.append(result[0])

  freelancerList=[idfreelancer]
  OffersId=[]
  nombreSkills=0  #to represent the nombre of skills that a particular freelancer have
  for skill in skills:
    if(skill in freelancerSkills) :
      freelancerList.append(1)
      nombreSkills += 1
    else :
      freelancerList.append(0)

  freelancerVector=np.array(freelancerList)
  SimilarityList=[]
  MissionMatrix=generateNonHiredMissionMatrix()
  for mission in MissionMatrix :
    shape=np.shape(mission)
    result=np.dot(freelancerVector[1:],mission[:][1:shape-2])
    SimilarityList.append(result[0][0])
  SelectedMissions=[]
  j=0
  for mission in MissionMatrix:
    if SimilarityList[j]>=nombreSkills*2/3 :
        SelectedMissions.append(mission)
    j+=1
  SortedSelectedMission=sorted(SelectedMissions,lambda mission:mission[shape(MissionMatrix)[1]-1],reverse=True)
  for item in SortedSelectedMission:
    OffersId.append(item[0])
  return OffersId
#recommmand freelancers to a particular mission
@app.route('/recommandfreelancer/<idMission>')
def recommandfreelancers(idMission):
  query='select technologies from mission where idMission=%s',idMission
  cursor.execute(query)
  technologie = cursor.fetchall()
  technologies = technologie[0][0].split("/")
  nombreSkillsDemanded=np.shape(technologies)
  freelancersId=[]
  MissionHeader = ["idMission"]
  MissionList=[idMission]
  for skill in skills:
      MissionHeader.append(skill)
      if skill in technologies:
        MissionList.append(1)
      else:
        MissionList.append(0)
  MissionInfoVector=np.array(MissionList)
  SimilarityList=[]
  for freelancer in freelancersMatrix :
    shape=np.shape(freelancer)
    result=np.dot(MissionInfoVector[1:],freelancer[:][1:shape-2])
    SimilarityList.append(result[0][0])
  SelectedFreelancers=[]
  j=0
  for freelancer in freelancersMatrix:
    if SimilarityList[j]>=nombreSkillsDemanded :
        SelectedFreelancers.append(freelancer)
    j+=1
  #We will replace the two last columns (rating,nbPorject) for each row in SelectedFreelancers by the factor calculated and put the result in a new matrix
  NewFreelancersMatrix=[]
  for freelancer in SelectedFreelancers:
      freelancerInfo=freelancer[0:shape(freelancer)-2]
      calculatedfactor=calculRatingNombreProjetFactor(freelancer[-2],freelancer[-1])
      freelancerInfo.append(calculatedfactor)
      NewFreelancersMatrix.append(freelancerInfo)
  SortedSelectedfreelancers = sorted(NewFreelancersMatrix, lambda factor: freelancer[shape(NewFreelancersMatrix)[1]-1], reverse=True)
  freelancersId=[freelancer[0] for freelancer in SortedSelectedfreelancers]
  return freelancersId
if __name__ == '__main__':
    app.run()
