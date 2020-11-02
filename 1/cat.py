class Cat:
    
    def __init__(self,name):
        self.name=name
    
    def greet(self,friend):
        print(f"""Hello, delicious friend {friend.name}! I, {self.name}, extend you my warmest greetings!
                Let us stroll through the fields together and terrify the tiny birds! Perhaps the human will accompany us?""")