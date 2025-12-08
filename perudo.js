// Constants
const MAX_PLAYERS = 4;
const DICES_PER_PLAYER = 5;
const DOUBT_ACTION = [-1, -1];
const JOLLY_FACE = 1;

// Generate all possible dice face counts for a given number of dices
const allPossibleDiceCountsTable = {};

function generateDiceCounts() {
    allPossibleDiceCountsTable[0] = [[0, 0, 0, 0, 0, 0]];
    
    for (let tot = 1; tot <= DICES_PER_PLAYER; tot++) {
        allPossibleDiceCountsTable[tot] = [];
        if (tot === 1) {
            for (let n = 0; n < 6; n++) {
                const fcs = [0, 0, 0, 0, 0, 0];
                fcs[n] = 1;
                allPossibleDiceCountsTable[tot].push(fcs);
            }
        } else {
            for (const prev of allPossibleDiceCountsTable[tot - 1]) {
                for (let n = 0; n < 6; n++) {
                    const fcs = [...prev];
                    fcs[n] += 1;
                    allPossibleDiceCountsTable[tot].push(fcs);
                }
            }
        }
    }
}

generateDiceCounts();

// Helper function to check if two arrays are equal
function arraysEqual(a, b) {
    return a.length === b.length && a.every((val, idx) => val === b[idx]);
}

// Check if a bid is legal
function isLegalBid(lastBid, newBid) {
    const [lastBidQuantity, lastBidFace] = lastBid;
    const [newBidQuantity, newBidFace] = newBid;

    if (lastBidQuantity === 0 && lastBidFace === 0) {
        return true;
    }

    if (arraysEqual(newBid, DOUBT_ACTION)) {
        return lastBidQuantity > 0;
    }

    let minBidQ = lastBidQuantity;
    let minBidF = lastBidFace;
    
    if (lastBidFace === JOLLY_FACE) {
        minBidQ = 2 * lastBidQuantity + 1;
        minBidF = 1;
    }

    if (newBidQuantity > minBidQ && newBidFace >= minBidF) {
        return true;
    }
    
    if (newBidQuantity >= minBidQ && newBidFace > minBidF) {
        return true;
    }

    if (newBidFace === JOLLY_FACE && newBidQuantity >= ((minBidQ % 2) + Math.floor(minBidQ / 2))) {
        return true;
    }

    return false;
}

// Get possible actions from a bid
function possibleActionsFromBid(totDices, lastBidQuantity, lastBidFace) {
    const actions = new Set();
    
    let minNextBidQ = Math.max(1, lastBidQuantity);
    let minNextBidF = Math.max(2, lastBidFace);
    
    // Doubt action only if there was a last bid
    if (lastBidQuantity > 0) {
        actions.add(JSON.stringify(DOUBT_ACTION));
    }

    // Last bid was jolly bid?
    if (lastBidFace === JOLLY_FACE) {
        minNextBidQ = 2 * lastBidQuantity + 1;
        minNextBidF = 2;
    }

    // Special jolly bid (only from second bid onwards)
    if (lastBidFace !== 0) {
        for (let bidQuantity = Math.floor(minNextBidQ / 2) + (minNextBidQ % 2); bidQuantity <= totDices; bidQuantity++) {
            actions.add(JSON.stringify([bidQuantity, JOLLY_FACE]));
        }
    }

    // Possible bids
    for (let bidQuantity = minNextBidQ; bidQuantity <= totDices; bidQuantity++) {
        for (let bidFace = minNextBidF; bidFace <= 6; bidFace++) {
            if (bidQuantity === minNextBidQ && bidFace === lastBidFace) {
                continue;
            }
            actions.add(JSON.stringify([bidQuantity, bidFace]));
        }
    }
    
    return Array.from(actions).map(a => JSON.parse(a));
}

// Player base class
class Player {
    constructor(name, nDices = DICES_PER_PLAYER) {
        this.name = name;
        this.nDices = nDices;
        this.history = [];
        this.newRound();
    }

    newRound() {
        const possibleCounts = allPossibleDiceCountsTable[this.nDices];
        this.dices = possibleCounts[Math.floor(Math.random() * possibleCounts.length)];
    }

    winBid() {
        // Small reward for winning one bid
    }

    loseDice() {
        this.nDices = Math.max(0, this.nDices - 1);
        if (this.nDices === 0) {
            this.lose();
        }
    }

    lose() {
        // Player lost
    }

    win() {
        // Player won
    }

    isAlive() {
        return this.nDices > 0;
    }

    makeAction(totalDices, lastBid) {
        // To be implemented by subclasses
        return null;
    }
}

// Human Player
class HumanPlayer extends Player {
    constructor(name, nDices = DICES_PER_PLAYER) {
        super(name, nDices);
        this.pendingAction = null;
        this.actionResolve = null;
    }

    makeAction(totalDices, lastBid) {
        return new Promise((resolve) => {
            this.actionResolve = resolve;
            // UI will call setAction when user makes a choice
        });
    }

    setAction(action) {
        if (this.actionResolve) {
            this.actionResolve(action);
            this.actionResolve = null;
        }
    }
}

// RoboPlayer base class
class RoboPlayer extends Player {
    getEstimateVector(totalDices) {
        const estimate = [0, 0, 0, 0, 0, 0];
        const otherDices = totalDices - this.nDices;
        
        for (let face = 1; face <= 6; face++) {
            if (face === JOLLY_FACE) {
                estimate[face - 1] = this.dices[0] + otherDices / 6;
            } else {
                estimate[face - 1] = this.dices[face - 1] + this.dices[0] + otherDices / 3;
            }
        }
        return estimate;
    }

    aggressiveAction(totalDices, lastBid) {
        const estimate = this.getEstimateVector(totalDices);
        const [lastBidQuantity, lastBidFace] = lastBid;
        const possibleActions = possibleActionsFromBid(totalDices, lastBidQuantity, lastBidFace);
        let bestAction = DOUBT_ACTION;
        let bestQuantity = -1;

        for (const action of possibleActions) {
            const [bidQuantity, bidFace] = action;
            if (arraysEqual(action, DOUBT_ACTION)) {
                continue;
            }
            if (bidQuantity <= estimate[bidFace - 1]) {
                if (bidQuantity > bestQuantity) {
                    bestQuantity = bidQuantity;
                    bestAction = action;
                }
            }
        }

        return bestAction;
    }

    conservativeAction(totalDices, lastBid) {
        const estimate = this.getEstimateVector(totalDices);
        const [lastBidQuantity, lastBidFace] = lastBid;
        const possibleActions = possibleActionsFromBid(totalDices, lastBidQuantity, lastBidFace);
        let bestAction = DOUBT_ACTION;
        let bestQuantity = totalDices + 1;

        for (const action of possibleActions) {
            const [bidQuantity, bidFace] = action;
            if (arraysEqual(action, DOUBT_ACTION)) {
                continue;
            }
            if (bidQuantity <= estimate[bidFace - 1]) {
                if (bidQuantity < bestQuantity) {
                    bestQuantity = bidQuantity;
                    bestAction = action;
                }
            }
        }

        return bestAction;
    }

    safeMinimalBid(totalDices, lastBid) {
        const estimate = this.getEstimateVector(this.nDices);
        const [lastBidQuantity, lastBidFace] = lastBid;
        const possibleActions = possibleActionsFromBid(totalDices, lastBidQuantity, lastBidFace);
        let bestAction = DOUBT_ACTION;
        let bestQuantity = totalDices + 1;

        for (const action of possibleActions) {
            const [bidQuantity, bidFace] = action;
            if (arraysEqual(action, DOUBT_ACTION)) {
                continue;
            }
            if (bidQuantity <= estimate[bidFace - 1]) {
                if (bidQuantity < bestQuantity) {
                    bestQuantity = bidQuantity;
                    bestAction = action;
                }
            }
        }

        return bestAction;
    }
}

// RoboPlayer variants
class AggressiveRoboPlayer extends RoboPlayer {
    makeAction(totalDices, lastBid) {
        return this.aggressiveAction(totalDices, lastBid);
    }
}

class ConservativeRoboPlayer extends RoboPlayer {
    makeAction(totalDices, lastBid) {
        return this.conservativeAction(totalDices, lastBid);
    }
}

class DoubterRoboPlayer extends RoboPlayer {
    makeAction(totalDices, lastBid) {
        return this.safeMinimalBid(totalDices, lastBid);
    }
}

// Game class
class Game {
    constructor(players) {
        this.players = players;
        this.nPlayers = players.length;
        this.startIdx = 0;
        this.currentRound = 0;
        this.lastBid = [0, 0];
        this.currentPlayerIdx = 0;
    }

    async playRound() {
        this.currentRound++;
        console.log("Starting a new round...");
        
        for (const p of this.players) {
            p.newRound();
        }
        
        const totalDices = this.players.reduce((sum, p) => sum + p.nDices, 0);
        this.lastBid = [0, 0];
        this.currentPlayerIdx = this.startIdx % this.nPlayers;

        while (true) {
            const currentPlayer = this.players[this.currentPlayerIdx];
            updateUI(this.currentPlayerIdx, this.lastBid, totalDices);
            
            let action;
            if (currentPlayer instanceof HumanPlayer) {
                action = await currentPlayer.makeAction(totalDices, this.lastBid);
            } else {
                // Add delay for robot players
                await new Promise(resolve => setTimeout(resolve, 1000));
                action = currentPlayer.makeAction(totalDices, this.lastBid);
            }

            console.log(`Player ${this.currentPlayerIdx}, ${currentPlayer.name} action:`, action);

            if (arraysEqual(action, DOUBT_ACTION)) {
                // Resolve doubt
                const bidderIdx = (this.currentPlayerIdx - 1 + this.nPlayers) % this.nPlayers;
                const bidder = this.players[bidderIdx];
                let actualCount = 0;
                const [q, f] = this.lastBid;
                
                for (const p of this.players) {
                    actualCount += p.dices[0]; // Jollies
                    if (f !== JOLLY_FACE) {
                        actualCount += p.dices[f - 1];
                    }
                }

                showMessage(`Actual count for face ${f === JOLLY_FACE ? 'Jolly' : f}: ${actualCount}`, 'info');

                if (actualCount >= q) {
                    // Bidder wins
                    this.startIdx = this.currentPlayerIdx;
                    currentPlayer.loseDice();
                    bidder.winBid();
                    showMessage(`${currentPlayer.name} doubted incorrectly! ${currentPlayer.name} loses a die.`, 'error');
                    
                    if (!currentPlayer.isAlive()) {
                        showMessage(`${currentPlayer.name} has been eliminated!`, 'error');
                        this.players.splice(this.currentPlayerIdx, 1);
                        this.nPlayers--;
                        // Adjust start index if needed
                        if (this.startIdx >= this.nPlayers) {
                            this.startIdx = 0;
                        }
                    }
                } else {
                    // Doubter wins
                    this.startIdx = bidderIdx;
                    bidder.loseDice();
                    currentPlayer.winBid();
                    showMessage(`${bidder.name} bid incorrectly! ${bidder.name} loses a die.`, 'success');
                    
                    if (!bidder.isAlive()) {
                        showMessage(`${bidder.name} has been eliminated!`, 'error');
                        const removeIdx = this.players.indexOf(bidder);
                        this.players.splice(removeIdx, 1);
                        this.nPlayers--;
                        // Adjust indices if needed
                        if (removeIdx < this.currentPlayerIdx) {
                            this.currentPlayerIdx--;
                        }
                        if (this.startIdx >= this.nPlayers) {
                            this.startIdx = 0;
                        }
                    }
                }
                break;
            } else {
                this.lastBid = action;
                const [q, f] = action;
                showMessage(`${currentPlayer.name} bids: ${q} ${f === JOLLY_FACE ? 'Jolly' : f}`, 'info');
                this.currentPlayerIdx = (this.currentPlayerIdx + 1) % this.nPlayers;
            }
        }
        
        if (this.nPlayers === 1) {
            const winner = this.players[0];
            winner.win();
            showMessage(`We have a winner: ${winner.name}!`, 'success');
            document.getElementById('gameEndActions').classList.remove('hidden');
            return true;
        }
        
        return false;
    }

    async playGame() {
        let gameOver = false;
        while (!gameOver) {
            gameOver = await this.playRound();
            if (!gameOver) {
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        }
    }
}

// Global game instance
let game = null;
let humanPlayer = null;

// UI Functions
function renderDice(diceValue) {
    const dice = document.createElement('div');
    dice.className = 'dice';
    
    if (diceValue === 1) {
        dice.classList.add('jolly');
        dice.textContent = '🎲';
    } else {
        const face = document.createElement('div');
        face.className = 'dice-face';
        face.style.display = 'grid';
        face.style.gridTemplateColumns = 'repeat(3, 1fr)';
        face.style.gridTemplateRows = 'repeat(3, 1fr)';
        face.style.width = '100%';
        face.style.height = '100%';
        face.style.padding = '5px';
        
        // Create dots based on dice value
        const dotPositions = {
            2: [[0,0], [2,2]],
            3: [[0,0], [1,1], [2,2]],
            4: [[0,0], [0,2], [2,0], [2,2]],
            5: [[0,0], [0,2], [1,1], [2,0], [2,2]],
            6: [[0,0], [0,1], [0,2], [2,0], [2,1], [2,2]]
        };
        
        const positions = dotPositions[diceValue] || [];
        
        for (let row = 0; row < 3; row++) {
            for (let col = 0; col < 3; col++) {
                const cell = document.createElement('div');
                cell.style.display = 'flex';
                cell.style.alignItems = 'center';
                cell.style.justifyContent = 'center';
                
                const hasDot = positions.some(p => p[0] === row && p[1] === col);
                if (hasDot) {
                    const dot = document.createElement('div');
                    dot.className = 'dot';
                    cell.appendChild(dot);
                }
                
                face.appendChild(cell);
            }
        }
        
        dice.appendChild(face);
    }
    
    return dice;
}

function renderPlayerDice(dices) {
    const container = document.getElementById('playerDice');
    container.innerHTML = '';
    
    for (let face = 0; face < 6; face++) {
        for (let count = 0; count < dices[face]; count++) {
            container.appendChild(renderDice(face + 1));
        }
    }
}

function updatePlayerStatus() {
    const statusDiv = document.getElementById('playerStatus');
    statusDiv.innerHTML = '';
    
    if (!game) return;
    
    game.players.forEach((player, idx) => {
        const playerDiv = document.createElement('div');
        playerDiv.className = 'player-status';
        if (idx === game.currentPlayerIdx) {
            playerDiv.classList.add('current');
        }
        playerDiv.textContent = `${player.name}: ${player.nDices} dice`;
        statusDiv.appendChild(playerDiv);
    });
}

function updateUI(currentPlayerIdx, lastBid, totalDices) {
    if (!game) return;
    
    updatePlayerStatus();
    
    // Always show player dice
    if (humanPlayer) {
        renderPlayerDice(humanPlayer.dices);
    }
    
    // Update last bid display
    const [q, f] = lastBid;
    if (q === 0 && f === 0) {
        document.getElementById('lastBid').textContent = 'No bid yet';
    } else {
        document.getElementById('lastBid').textContent = `Last bid: ${q} ${f === JOLLY_FACE ? 'Jolly' : f}`;
    }
    
    // Show bid section only for human player's turn
    if (currentPlayerIdx === 0 && humanPlayer) {
        document.getElementById('bidSection').classList.remove('hidden');
        document.getElementById('waitingMessage').classList.add('hidden');
        document.getElementById('doubtButton').disabled = (q === 0 && f === 0);
    } else {
        document.getElementById('bidSection').classList.add('hidden');
        document.getElementById('waitingMessage').classList.remove('hidden');
    }
}

function showMessage(text, type) {
    const messageArea = document.getElementById('messageArea');
    const message = document.createElement('div');
    message.className = `message ${type}`;
    message.textContent = text;
    messageArea.innerHTML = '';
    messageArea.appendChild(message);
}

function makeBid() {
    if (!humanPlayer || !humanPlayer.actionResolve) return;
    if (!game) return;
    
    const quantity = parseInt(document.getElementById('bidQuantity').value);
    const face = parseInt(document.getElementById('bidFace').value);
    
    if (isNaN(quantity) || quantity < 1) {
        showMessage('Please enter a valid quantity', 'error');
        return;
    }
    
    const newBid = [quantity, face];
    const actualLastBid = game.lastBid || [0, 0];
    
    if (!isLegalBid(actualLastBid, newBid)) {
        showMessage('Illegal bid! Please check the rules.', 'error');
        return;
    }
    
    humanPlayer.setAction(newBid);
    document.getElementById('bidQuantity').value = '';
}

function doubtBid() {
    if (!humanPlayer || !humanPlayer.actionResolve) return;
    
    const lastBidText = document.getElementById('lastBid').textContent;
    if (lastBidText === 'No bid yet') {
        showMessage('Cannot doubt when there is no bid', 'error');
        return;
    }
    
    humanPlayer.setAction(DOUBT_ACTION);
    showMessage('You doubt the last bid!', 'info');
}

function resetGame() {
    document.getElementById('gameEndActions').classList.add('hidden');
    document.getElementById('gameSetup').classList.remove('hidden');
    document.getElementById('gameArea').classList.add('hidden');
    game = null;
    humanPlayer = null;
}

function startNewGame() {
    const opponentType = document.getElementById('opponentSelect').value;
    
    humanPlayer = new HumanPlayer('You');
    let opponent;
    
    switch (opponentType) {
        case 'aggressive':
            opponent = new AggressiveRoboPlayer('Aggressive Bot');
            break;
        case 'conservative':
            opponent = new ConservativeRoboPlayer('Conservative Bot');
            break;
        case 'doubter':
            opponent = new DoubterRoboPlayer('Doubter Bot');
            break;
    }
    
    game = new Game([humanPlayer, opponent]);
    
    document.getElementById('gameSetup').classList.add('hidden');
    document.getElementById('gameArea').classList.remove('hidden');
    document.getElementById('gameEndActions').classList.add('hidden');
    document.getElementById('messageArea').innerHTML = '';
    
    game.playGame();
}

// Allow Enter key to submit bid
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('bidQuantity').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            makeBid();
        }
    });
});
