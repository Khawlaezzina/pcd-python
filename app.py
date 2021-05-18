import math

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, jsonify
import pymysql
import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)
CORS(app)

conn=pymysql.connect(host='localhost',user='root',password='',database='pcd',autocommit=True)
cursor=conn.cursor()
query1='select id,rating from freelancer'
query2='select id,average_payment,technologies,id_client from missions where hired=0' #to change
query3='select * from client'
query4='select name from skills'
query5='select * from skilled'


nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()
#tfId Vectorization example
def tfId():
  ds = pd.read_csv("Data/skills.csv")
  print(ds)
  countvectorizer = CountVectorizer(analyzer='word', stop_words='english')
  count_wm = countvectorizer.fit_transform(ds)
  count_tokens = countvectorizer.get_feature_names()
  df_countvect = pd.DataFrame(data=count_wm.toarray(), index=['Skills'], columns=count_tokens)
  print("Count Vectorizer\n")
  print(df_countvect)
#defining a list containing all the name skills presented on the table skills
def generateSkillsList():
  try :
    cursor.execute(query4)
    rows=cursor.fetchall()
    skills=[]
    for row in rows:
      skills.append(row[0])
    print(skills)
    return skills
  except:
    print("error dans generateSkillsList")

#defining a matrix containing all non hired missions on each row we find [idMission,skills,ratingClient]
def generateNonHiredMissionMatrix():
  try :
    skills=generateSkillsList()
    cursor.execute(query2)
    missions=cursor.fetchall()

    VectorMissionHeader=["idMission"]
    for skill in skills :
      VectorMissionHeader.append(skill)
    VectorMissionHeader.append('ratingClient')
    MissionMatrix1=[]
    for mission in missions :
      MissionVector=[mission[0]]
      technologies=mission[2].split("/") #I should verify the number of column technologies on table mission
      for skill in skills:
         if skill in technologies :
           MissionVector.append(1)
         else :
          MissionVector.append(0)
      query6='select rating from client where id in (select id_client from missions where id=%s)'
      cursor.execute(query6,mission[3])
      result=cursor.fetchall()
      rating=result[0][0]

      MissionVector.append(rating)
      MissionMatrix1.append(MissionVector)

    MissionMatrix=np.array(MissionMatrix1)
    print(VectorMissionHeader)
    print(MissionMatrix)
    return MissionMatrix
  except:
    print("error in generateNonHiredMissionMatrix")
#defining a freelancers Matrix containg on each row [idFreelancer,skills,rating,nbreprojets]
def generateFreelancersMatrix():
  try:
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
      query = 'select skills.name from skills where id_skill in (select skill_id from skilled where freelancer_id=%s)'
      cursor.execute(query,freelancer[0])
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

      embeddedquery='select count(id) from missions where completed=true and id_freelancer=%s' #to change
      cursor.execute(embeddedquery,freelancer[0])
      result2=cursor.fetchall()

      nombreProjets=result2[0][0]

      freelancerList.append(nombreProjets)
      freelancersMatrix1.append(freelancerList)
    print(Freelancerheaders)
    print(freelancersMatrix1)
    return freelancersMatrix1
  except:
    print("Error in generate Freelancers Matrix")


def calculRatingNombreProjetFactor(rating,nombreProjet,similarity):
  return 0.37*rating+0.3*nombreProjet+0.33*similarity
def calculOfferId(Matrix):
  OffersId=[]
  for line in Matrix:

    OffersId.append(int(line[0]))
  return OffersId
@app.route('/',methods=['GET'])
def hello_world():
    return 'Hello World!'

#recommand offers to a particular freelancer
@app.route('/recommandOffer/<idfreelancer>',methods=['GET','POST'])
def recommandOffers(idfreelancer):
  try :
    skills = generateSkillsList()

    query='select skills.name from skills where id_skill in (select skill_id from skilled where freelancer_id=%s)'
    cursor.execute(query,idfreelancer)
    result1 = cursor.fetchall()
    print(result1)
    freelancerSkills = []
    headers = ['idfreelancer']

    for skill in skills :
      headers.append(skill)
    for result in result1:
      freelancerSkills.append(result[0])

    freelancerList=[idfreelancer]
    nombreSkills=0  #to represent the nombre of skills that a particular freelancer have
    for skill in skills:
      if(skill in freelancerSkills) :
        freelancerList.append(1)
        nombreSkills += 1
      else :
        freelancerList.append(0)
    freelancerVector=np.array(freelancerList)
    print(headers)
    print(freelancerVector)
    newFreelancerVector=[]
    for i in freelancerVector:
      newFreelancerVector.append(int(i))
    SimilarityList=[]
    MissionMatrix=generateNonHiredMissionMatrix()

    for mission in MissionMatrix :
      shape=np.shape(mission)
      nombreColonne=shape[0]

      result=np.dot(newFreelancerVector[1:],mission[1:nombreColonne-1])
      #result=cosine_similarity(newFreelancerVector[1:],mission[1:nombreColonne-1])
      SimilarityList.append(result)
    #Similarity=cosine_similarity(newFreelancerVector.toarray(),MissionMatrix,True)
    #print(Similarity)
    SelectedMissions=[]
    j=0
    for mission in MissionMatrix:
      if SimilarityList[j]>=1.0 :
        SelectedMissions.append(mission.tolist())
      j+=1
    print(SelectedMissions)

    index=len(SelectedMissions[0])-1

    SortedSelectedMission=sorted(SelectedMissions,key=lambda mission: mission[index],reverse=True)
    print(SortedSelectedMission)
    OffersId=calculOfferId(SortedSelectedMission)
    print(OffersId)
    dictionnaire= {
        'titre':'les missions recommandees pour un freelancer',
        'liste':OffersId
    }
    return jsonify(dictionnaire)
  except:
    print("Error in recommand offers for a freelancer")
    return jsonify("bonjour")
  finally:
    cursor.close()
    conn.close()
#recommmand freelancers to a particular mission
@app.route('/recommandfreelancer/<idMission>',methods=['GET','POST'])
def recommandfreelancers(idMission):
  try:
    query='select technologies from missions where id=%s'
    cursor.execute(query,idMission)
    technologie = cursor.fetchall()
    print(technologie)
    skills = generateSkillsList()
    print(skills)
    technologies = technologie[0][0].split("/")
    print(technologies)
    numbreSkillsDemanded=np.shape(technologies)[0]
    print(numbreSkillsDemanded)
    MissionHeader = ["idMission"]
    MissionList=[int(idMission)]
    for skill in skills:
      MissionHeader.append(skill)
      if skill in technologies:
        MissionList.append(int(1))
      else:
        MissionList.append(int(0))
    MissionInfoVector=np.array(MissionList)
    print(MissionHeader)
    print(MissionInfoVector)
    SimilarityList=[]
    freelancersMatrix=np.array(generateFreelancersMatrix())
    print(freelancersMatrix)
    for freelancer in freelancersMatrix :

      shape=np.shape(freelancer)[0]-2

      result=np.dot(MissionInfoVector[1:],freelancer[1:shape])

      SimilarityList.append(int(result))
    print(freelancersMatrix)
    i=0
    Selectedfreelancers=[]
    NewSimilarityList=[]
    for freelancer in freelancersMatrix:
      if SimilarityList[i]>=1.0 :
        Selectedfreelancers.append(freelancer)
        NewSimilarityList.append(SimilarityList[i])
      i+=1
  #We will replace the two last columns (rating,nbPorject,SimilarityList) for each row in SelectedFreelancers by the factor calculated and put the result in a new matrix
    NewFreelancersMatrix=[]
    NewHeaders=['idFreelancer']
    for skill in skills :
      NewHeaders.append(skill)
    NewHeaders.append('CalculatedFactor')
    j=0
    for freelancer in Selectedfreelancers:


      print(len(freelancer))
      freelancerInfo=freelancer[0:len(freelancer)-2].tolist()
      print(freelancerInfo)

      calculatedfactor=calculRatingNombreProjetFactor(freelancer[-2],freelancer[-1],SimilarityList[j])
      freelancerInfo.append(calculatedfactor)
      NewFreelancersMatrix.append(freelancerInfo)
      j+=1
    print(NewHeaders)
    print(NewFreelancersMatrix)
    index=len(NewFreelancersMatrix[0])-1
    SortedSelectedfreelancers = sorted(NewFreelancersMatrix,key=lambda freelancer: freelancer[index], reverse=True)
    freelancersId=[int(freelancer[0]) for freelancer in SortedSelectedfreelancers]
    print(freelancersId)
    dictionnaire= {
        'titre':'les freelancers recommandees pour cette mission',
        'liste':freelancersId
    }
    return jsonify(dictionnaire)

  except:
    print("Error on recommand freealncers for an offer")
    return 'Error'
  finally:
    cursor.close()
    conn.close()
#update client rating
@app.route('/updateClientRating/<idClient>',methods=['GET','POST'])
def updateClientRating(idClient):
  try:
    reviews='select id_freelancer,comment_freelancer from reviews where id_client=%s'
    cursor.execute(reviews,idClient)
    results=cursor.fetchall()
    clientReviews=[]
    idFreelancers=[]
    scores=[]
    for result in results:
      clientReviews.append(result[1])
      idFreelancers.append(result[0])
    print(clientReviews)
    print(idFreelancers)
    if clientReviews==[] :
      rating=0
    else:
      for review in clientReviews:
        print(analyzer.polarity_scores(review)['compound'])
        scores.append(analyzer.polarity_scores(review)['compound'])
      print(scores)
      scoremoyen=np.mean(scores)
      rating=math.ceil(scoremoyen/5)   #calculer le nombre des étoiles à attribuer sur 5 pour un client
      print(rating)
    update='update client set rating=%s where id=%s'
    val=(rating,int(idClient))
    cursor.execute(update,val)
    print('Doneee !!!!')
    dictionnaire= {
      'New rating':rating,
      'freelancersId':idFreelancers,
      'Reviews ':clientReviews
    }
    return jsonify(dictionnaire)
  except:
      print('error in update client rating')
      return 'error'
  finally:
    cursor.close()
    conn.close()
#update freelancer rating
@app.route('/upadateFreelancerRating/<idFreelancer>',methods=['GET','POST'])
def updatefreelancerrating(idFreelancer):
  try :
    reviews='select id_client,comment_client from reviews where id_freelancer=%s'
    cursor.execute(reviews,idFreelancer)
    results=cursor.fetchall()
    print(results)
    freelancerReviews=[]
    idClients=[]
    scores=[]
    for result in results:
      freelancerReviews.append(result[1])
      idClients.append(result[0])
    print(freelancerReviews)
    print(idClients)
    if freelancerReviews==[]:
      rating=0
    else:
      for review in freelancerReviews:
        print(analyzer.polarity_scores(review)['compound'])
        scores.append(analyzer.polarity_scores(review)['compound'])
      print(scores)
      MoyScore=np.mean(scores)
      print(MoyScore)
      rating=math.ceil(MoyScore/5)  #calculer le nombre des étoiles à attribuer sur 5 pour le freelancer
      print(rating)
    update='update freelancer set rating=%s where id=%s'
    val=(rating,int(idFreelancer))
    cursor.execute(update,val)
    #conn.commit()
    print('Hello')
    dictionnaire = {
      'Reviews': freelancerReviews,
      'idClients':idClients,
      'rating':rating
    }
    print('Update done successfully')
    return jsonify(dictionnaire)

  except:
    print('Error on update freelancer rating')
    return jsonify('ERROR')
  finally:
    cursor.close()
    conn.close()
if __name__ == '__main__':
    app.run()

