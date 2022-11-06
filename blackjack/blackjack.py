import random

"""
0: stick, 1: hit
"""

#deck = [f"{i}" for i in range(1, 11)] + ["10"]*3
deck = [i for i in range(1, 11)] + [10]*3
deck = deck * 4

verbose = False 
showDealerHand = False

def drawCard(deck):
    randPos = random.randint(0, len(deck)-1)
    card = deck.pop(randPos)
    return card

# deal with ordering problems and two digit numbers
def cardsToStr(cards, dealerCard):
    cardString = "  " # fix: there needs to be 4 characters for the top 2
    """
    if dealerCard == 10:
        cardString = f"A "
    else:
        cardString = f"{dealerCard} "
    """

    #cardString += str(cards.count(1)) + " "
    #cardString += str(sum(cards))
    #cards = sorted(cards)
    for card in cards:
        if card == 10:
            cardString += "A"
        else:
            cardString += f"{card}"
    return cardString

def getMaxSum(cards):
    cardsSum = 0
    for card in cards:
        if card == 1:
            cardsSum += 11
        else:
            cardsSum += card
    return cardsSum

def findBetterSum(sumA, sumB):
    if sumA > 21:
        return sumB
    if sumB > 21:
        return sumA
    return max(sumA, sumB)

def getBestSum(cards, runningSum=0):
    if runningSum > 21: return 22
    if len(cards) == 0: return runningSum
    nextCard = cards.pop()
    if nextCard == 1:
        return findBetterSum(getBestSum(cards.copy(), runningSum+1), getBestSum(cards.copy(), runningSum+11))
    return getBestSum(cards.copy(), runningSum+nextCard)

# randomly hit or stick
def randomPolicy(cards, dealerCard):
    if getMaxSum(cards) < 12:
        return 1
    return random.randint(0,1)

def getMonteCarloPolicy(valueFunction, eps=0.05):
    def monteCarloPolicy(cards, dealerCard):
        if getMaxSum(cards) < 12:
            return 1
        """
        if random.uniform(0, 1) < eps:
            return random.randint(0,1)
        """
        cardsString = cardsToStr(cards, dealerCard)
        if cardsString in valueFunction:
            if 0 in valueFunction[cardsString] and 1 in valueFunction[cardsString]:
                if valueFunction[cardsString][0] > valueFunction[cardsString][1]:
                    if random.uniform(0, 1) < eps:
                        return 1
                    else:
                        return 0
                if random.uniform(0, 1) < eps:
                    return 0
                else:
                    return 1
            else:
                return random.randint(0,1)
        return random.randint(0,1)
    return monteCarloPolicy

def humanPolicy(cards, dealerCard):
    if len(cards) == 2: print("The dealer has:", dealerCard)
    print("You have:", cards)
    return int(input("Enter 1 to hit or 0 to stick: "))

# sticks on any sum geq 17
def getDealerPolicy(threshold):
    def dealerPolicy(cards):
        if getMaxSum(cards) >= threshold: return 0
        return 1
    return dealerPolicy

def updateHistory(history, episode, actions, score):
    # {episode: {action: [scores]}}
    if len(actions) < 1: return
    if episode[0:4] not in history:
        history[episode[0:4]] = {}
    if actions[0] not in history[episode[0:4]]:
        history[episode[0:4]][actions[0]] = []
    history[episode[0:4]][actions[0]].append(score)
    for i in range(1, len(actions)):
        if episode[0:4+i] not in history:
            history[episode[0:4+i]] = {}
        if actions[i] not in history[episode[0:4+i]]:
            history[episode[0:4+i]][actions[i]] = []
        history[episode[0:4+i]][actions[0+i]].append(score)

def averageHistory(history):
    valueFunction = {}
    for history, action_dict in history.items():
        valueFunction[history] = {}
        for action, scores in action_dict.items():
            valueFunction[history][action] = sum(scores)/len(scores)
    return valueFunction

def game(playerPolicy, dealerPolicy):
    if verbose: print("The game has started!")
    _deck = deck.copy()

    dealerCards = [drawCard(_deck), drawCard(_deck)] # the top card is public
    playerCards = [drawCard(_deck), drawCard(_deck)]
    playerActions = []
    if verbose:
        print("The dealer's cards:")
        print(dealerCards)
        print("The player's cards:")
        print(playerCards)

    # player first
    while sum(playerCards) < 21:
        playerChoice = playerPolicy(playerCards, dealerCards[0])
        playerActions.append(playerChoice)
        if verbose: print("The player picked:", playerChoice)
        if playerChoice == 1:
            playerCards.append(drawCard(_deck))
            if verbose:
                print("The player's cards:")
                print(playerCards)
        else:
            break

    if sum(playerCards) > 21: return cardsToStr(playerCards, dealerCards[0]), playerActions, -1

    # dealer second
    while sum(dealerCards) < 21:
        dealerChoice = dealerPolicy(dealerCards)
        if verbose: print("The dealer picked:", dealerChoice)
        if dealerChoice == 1:
            dealerCards.append(drawCard(_deck))
            if verbose:
                print("The dealer's cards:")
                print(dealerCards)
        else:
            break

    if showDealerHand: print("The dealer's hand:", dealerCards)
    if sum(dealerCards) > 21: return cardsToStr(playerCards, dealerCards[0]), playerActions, 1

    bestPlayerSum = getBestSum(playerCards.copy())
    bestDealerSum = getBestSum(dealerCards.copy())
    if bestPlayerSum == bestDealerSum: return cardsToStr(playerCards, dealerCards[0]), playerActions, 0
    if bestPlayerSum > bestDealerSum: return cardsToStr(playerCards, dealerCards[0]), playerActions, 1
    return cardsToStr(playerCards, dealerCards[0]), playerActions, -1

def policyIterationExperiment(eps=0.1):
    numGames = 100000
    threshold = 17
    history = {} # episode: {action: score}
    aveScore = 0
    for i in range(numGames):
        episode, actions, score = game(randomPolicy, getDealerPolicy(threshold))
        if verbose:
            print(episode)
            print(actions)
            print(score)
        aveScore += score/numGames
        updateHistory(history, episode, actions, score)

    print(aveScore)
    valueFunction = averageHistory(history)
    print("Found the Initial Value Function")

    for j in range(10):
        aveScore = 0
        smartPlayerPolicy = getMonteCarloPolicy(valueFunction, eps=eps)
        for i in range(numGames):
            episode, actions, score = game(smartPlayerPolicy, getDealerPolicy(threshold))
            aveScore += score/numGames
            updateHistory(history, episode, actions, score)
        print("Round", j, ":", aveScore)
        valueFunction = averageHistory(history)

    print("Doing a final evaluation")
    aveScore = 0
    smartPlayerPolicy = getMonteCarloPolicy(valueFunction, eps=0)
    for i in range(numGames):
        episode, actions, score = game(smartPlayerPolicy, getDealerPolicy(threshold))
        aveScore += score/numGames
        updateHistory(history, episode, actions, score)
    print("Score:", aveScore)


if __name__ == "__main__":
    policyIterationExperiment()
    # random policy to get a good value function
    """
    numGames = 100000
    threshold = 17
    history = {} # episode: {action: score}
    aveScore = 0
    for i in range(numGames):
        episode, actions, score = game(randomPolicy, getDealerPolicy(threshold))
        if verbose:
            print(episode)
            print(actions)
            print(score)
        aveScore += score/numGames
        updateHistory(history, episode, actions, score)

    print(aveScore)
    valueFunction = averageHistory(history)
    print("Found the Value Function")

    smartPlayerPolicy = getMonteCarloPolicy(valueFunction)
    aveScore = 0
    for i in range(numGames):
        episode, actions, score = game(smartPlayerPolicy, getDealerPolicy(threshold))
        aveScore += score/numGames
    print(aveScore)
    """

    """
    for i in range(10):
        episode, actions, score = game(humanPolicy, getDealerPolicy(17))
        if score == -1:
            print("you lost :( with a hand:", episode)
        elif score == 0:
            print("you drew")
        else:
            print("you won!!")
        print()
    """

            



