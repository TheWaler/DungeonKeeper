import os
from langchain import OpenAI
from keeper import Keeper
from archive import Archive, Adventure
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredHTMLLoader

test_adventure = Adventure("Name of Adventure", "location/of/adventure.md")

def main():
    load_dotenv() 
    verbose = os.getenv("VERBOSE") == "TRUE"
    db_path = os.getenv("DB_PATH")

    print("**********************************************")
    print("*                                            *")
    print("*  Welcome to the Dungeon Keeper!            *")
    print("*                                            *")
    print("*  Attempting to load model...               *")
    model = OpenAI()
    print("* Model loaded!                              *")
    print("*                                            *")
    print("*  Attempting to load the archives...        *")
    print("*                                            *")
    archive = Archive(db_path=db_path, model=model, verbose=verbose)
    print("* Archives loaded!                           *")
    print("*                                            *")
    print("*  Attempting to load the adventure...       *")
    archive.save_adventure(test_adventure)
    print("* Adventure loaded!                          *")
    print("*  Attempting to call the keeper...          *")
    dungeoun_keeper = Keeper(generation_model=model, verbose=verbose, archive=archive, adventure=test_adventure)
    print("*                                            *")
    print("* We are ready to begin!                     *")
    
    while True:
        question = input("Ask a question: ")
        answer = dungeoun_keeper.answer(question)
        print(answer)

if __name__ == '__main__':
    main()