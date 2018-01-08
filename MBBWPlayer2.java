import java.util.ArrayList;
import java.util.Random;

/**
 * GreedyMCPlayer - a simple, greedy Monte Carlo implementation of the player interface for PokerSquares.
 * For each possible play, continues greedy play with random possible card draws to a given depth limit 
 * (or game end).  Having sampled trajectories for all possible plays, the GreedyMCPlayer then selects the
 * play yielding the best average scoring potential in such Monte Carlo simulation.
 * 
 * Disclaimer: This example code is not intended as a model of efficiency. (E.g., patterns from Knuth's Dancing Links
 * algorithm (DLX) can provide faster legal move list iteration/deletion/restoration.)  Rather, this example
 * code illustrates how a player could be constructed.  Note how time is simply managed so as to not run out the play clock.
 * 
 * Author: Todd W. Neller
 */
public class MBBWPlayer2 implements PokerSquaresPlayer {
	
	private final int SIZE = 5; // number of rows/columns in square grid
	private final int NUM_POS = SIZE * SIZE; // number of positions in square grid
	private final int NUM_CARDS = Card.NUM_CARDS; // number of cards in deck
	private Random random = new Random(); // pseudorandom number generator for Monte Carlo simulation 
	private int[] plays = new int[NUM_POS]; // positions of plays so far (index 0 through numPlays - 1) recorded as integers using row-major indices.
	// row-major indices: play (r, c) is recorded as a single integer r * SIZE + c (See http://en.wikipedia.org/wiki/Row-major_order)
	// From plays index [numPlays] onward, we maintain a list of yet unplayed positions.
	private int numPlays = 0; // number of Cards played into the grid so far
	private PokerSquaresPointSystem system; // point system
	private int depthLimit = 2; // default depth limit for Greedy Monte Carlo (MC) play
	private Card[][] grid = new Card[SIZE][SIZE]; // grid with Card objects or null (for empty positions)
	private Card[] simDeck = Card.getAllCards(); // a list of all Cards. As we learn the index of cards in the play deck,
	                                             // we swap each dealt card to its correct index.  Thus, from index numPlays 
												 // onward, we maintain a list of undealt cards for MC simulation.
	private int[][] legalPlayLists = new int[NUM_POS][NUM_POS]; // stores legal play lists indexed by numPlays (depth)
	// (This avoids constant allocation/deallocation of such lists during the greedy selections of MC simulations.)
	double initEpsilon = 0.1, epsilon = initEpsilon;
	int numIter = 10000000;
	private double[][] values = new double [SIZE][SIZE];
	private double[][] visits  = new double [SIZE][SIZE];
	/**
	 * Create a Greedy Monte Carlo player that simulates greedy play to depth 2.
	 */
	public MBBWPlayer2() {
		for (int i = 0; i < SIZE; i++){
			for (int j = 0; j < SIZE; j++){
				visits[i][j] = 1;
			}
		}
	}
	
	/**
	 * Create a Greedy Monte Carlo player that simulates greedy play to a given depth limit.
	 * @param depthLimit depth limit for random greedy simulated play
	 */
	public MBBWPlayer2(int depthLimit) {
		for (int i = 0; i < SIZE; i++){
			for (int j = 0; j < SIZE; j++){
				visits[i][j] = 1;
			}
		}
		this.depthLimit = depthLimit;
	}
	
	/* (non-Javadoc)
	 * @see PokerSquaresPlayer#init()
	 */
	@Override
	public void init() { 
		// clear grid
		for (int row = 0; row < SIZE; row++)
			for (int col = 0; col < SIZE; col++)
				grid[row][col] = null;
		// reset numPlays
		numPlays = 0;
		// (re)initialize list of play positions (row-major ordering)
		for (int i = 0; i < NUM_POS; i++)
			plays[i] = i;
	}

	/* (non-Javadoc)
	 * @see PokerSquaresPlayer#getPlay(Card, long)
	 */
	@Override
	public int[] getPlay(Card card, long millisRemaining) {
		/*
		 * With this algorithm, the player chooses the legal play that has the highest expected score outcome.
		 * This outcome is estimated as follows:
		 *   For each move, many simulated greedy plays to the set depthLimit are performed and the (sometimes
		 *     partially-filled) grid is scored.
		 *   For each greedy play simulation, random undrawn cards are drawn in simulation and the greedy player
		 *     picks a play position that maximizes the score (breaking ties randomly).
		 *   After many such plays, the average score per simulated play is computed.  The play with the highest 
		 *     average score is chosen (breaking ties randomly).   
		 */
		
		// match simDeck to actual play event; in this way, all indices forward from the card contain a list of 
		//   undealt Cards in some permutation.
		int cardIndex = numPlays;
		while (!card.equals(simDeck[cardIndex]))
			cardIndex++;
		simDeck[cardIndex] = simDeck[numPlays];
		simDeck[numPlays] = card;
		
		if (numPlays < 24) { // not the forced last play
			// compute average time per move evaluation
			int remainingPlays = NUM_POS - numPlays; // ignores triviality of last play to keep a conservative margin for game completion
			long millisPerPlay = millisRemaining / remainingPlays; // dividing time evenly with future getPlay() calls
			long millisPerMoveEval = millisPerPlay / remainingPlays; // dividing time evenly across moves now considered
			// copy the play positions (row-major indices) that are empty
			System.arraycopy(plays, numPlays, legalPlayLists[numPlays], 0, remainingPlays);
			double maxAverageScore = Double.NEGATIVE_INFINITY; // maximum average score found for moves so far
			ArrayList<Integer> bestPlays = new ArrayList<Integer>(); // all plays yielding the maximum average score 
			for (int i = 0; i < remainingPlays; i++) { // for each legal play position
				int play = legalPlayLists[numPlays][i];
				long startTime = System.currentTimeMillis();
				long endTime = startTime + millisPerMoveEval; // compute when MC simulations should end
				makePlay(card, play / SIZE, play % SIZE);  // play the card at the empty position
				int simCount = 0;
				int scoreTotal = 0;
				while (System.currentTimeMillis() < endTime || simCount == numIter) { // perform as many MC simulations as possible through the allotted time
					// Perform a Monte Carlo simulation of greedy play to the depth limit or game end, whichever comes first.
					epsilon = initEpsilon * (numIter - i) / numIter;
					scoreTotal += MCSim(play);  // accumulate MC simulation scores
					simCount++; // increment count of MC simulations
				}
				undoPlay(); // undo the play under evaluation
				// update (if necessary) the maximum average score and the list of best plays
				double averageScore = (double) scoreTotal / simCount;
				if (averageScore >= maxAverageScore) {
					if (averageScore > maxAverageScore)
						bestPlays.clear();
					bestPlays.add(play);
					maxAverageScore = averageScore;
				}
			}
			int bestPlay = bestPlays.get(random.nextInt(bestPlays.size())); // choose a best play (breaking ties randomly)
			// update our list of plays, recording the chosen play in its sequential position; all onward from numPlays are empty positions
			int bestPlayIndex = numPlays;
			while (plays[bestPlayIndex] != bestPlay)
				bestPlayIndex++;
			plays[bestPlayIndex] = plays[numPlays];
			plays[numPlays] = bestPlay;
		}
		int[] playPos = {plays[numPlays] / SIZE, plays[numPlays] % SIZE}; // decode it into row and column
		makePlay(card, playPos[0], playPos[1]); // make the chosen play (not undoing this time)
		return playPos; // return the chosen play
	}

	double MCSim(int p) {
		// If the game is over, return the resulting value.
		if (NUM_POS == numPlays) {
			return system.getScore(grid);
		}
		// Choose best estimated move or a random move with a given probability. (epsilon-greedy)

		double score = Integer.MIN_VALUE;
		double maxScore = Integer.MIN_VALUE;
			// generate a random card draw
			int c = random.nextInt(NUM_CARDS - numPlays) + numPlays;
			Card card = simDeck[c];
			// iterate through legal plays and choose the best greedy play (see similar approach in getPlay)
			int remainingPlays = NUM_POS - numPlays;
			System.arraycopy(plays, numPlays, legalPlayLists[numPlays], 0, remainingPlays);
			maxScore = Integer.MIN_VALUE;
			ArrayList<Integer> bestPlays = new ArrayList<Integer>();
			for (int i = 0; i < remainingPlays; i++) {
				int play = legalPlayLists[numPlays][i];
				score = values[play/SIZE][play%SIZE] / visits[play/SIZE][play%SIZE];
				if (score >= maxScore) {
					if (score > maxScore)
						bestPlays.clear();
					bestPlays.add(play);
					maxScore = score;
				}
			}
			int bestPlay = bestPlays.get(random.nextInt(bestPlays.size()));
		if (random.nextDouble() < epsilon)
			 bestPlay = legalPlayLists[numPlays][random.nextInt(remainingPlays)];

		// Make the play, simulated the roll (if needed), and recursively call MCSim to get the terminal value.
		makePlay(card, bestPlay / SIZE, bestPlay % SIZE);
		double value = 0;
		value = MCSim(bestPlay);
		undoPlay();
		// Update visitCount pWinRoll/Hold, and pWin

			visits[p / SIZE][p % SIZE]++;
			values[p / SIZE][p % SIZE] += value;

		return value;
	}
	
	
	public void makePlay(Card card, int row, int col) {
		// match simDeck to event
		int cardIndex = numPlays;
		while (!card.equals(simDeck[cardIndex]))
			cardIndex++;
		simDeck[cardIndex] = simDeck[numPlays];
		simDeck[numPlays] = card;
		
		// update plays to reflect chosen play in sequence
		grid[row][col] = card;
		int play = row * SIZE + col;
		int j = 0;
		while (plays[j] != play)
			j++;
		plays[j] = plays[numPlays];
		plays[numPlays] = play;
		
		// increment the number of plays taken
		numPlays++;
	}

	public void undoPlay() { // undo the previous play
		numPlays--;
		int play = plays[numPlays];
		grid[play / SIZE][play % SIZE] = null;	
	}

	/* (non-Javadoc)
	 * @see PokerSquaresPlayer#setPointSystem(PokerSquaresPointSystem, long)
	 */
	@Override
	public void setPointSystem(PokerSquaresPointSystem system, long millis) {
		this.system = system;
	}

	/* (non-Javadoc)
	 * @see PokerSquaresPlayer#getName()
	 */
	@Override
	public String getName() {
		return "MBBWPlayer2";
	}

	public static void main(String[] args) {
		PokerSquaresPointSystem system = PokerSquaresPointSystem.getRandomPointSystem();
		System.out.println(system);
//		new PokerSquares(new MBBWPlayer(), system).play(); // play a single game
//		System.out.println("******");
		new PokerSquares(new MBBWPlayer2(3), system).play(); // play a single game
		
//		ArrayList<PokerSquaresPlayer> players = new ArrayList<PokerSquaresPlayer>();
//		players.add(new MBBWPlayer());
//		players.add(new MBBWPlayer2(3));
//		ArrayList<PokerSquaresPointSystem> systems = new ArrayList<PokerSquaresPointSystem>();
//		systems.add(system);
//		PokerSquares.playTournament(players,systems,5, 2);
	}


}
