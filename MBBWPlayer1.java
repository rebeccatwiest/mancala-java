/**
 * Tneller2MancalaPlayer - My simple implementation of a mancala
 * player which seeks to distribute search computation through
 * iterative-deepening.
 *
 * @author Todd Neller
 * @version 1.1 */
public class MBBWMancalaPlayer extends TnellerMancalaPlayer {

    /**
     * Choose a move for the given game situation given play time
     * remaining.  */
    public int chooseMove(MancalaNode node, long timeRemaining) {
	final int TOTAL_PIECES = 48;
	final long TOTAL_TIME = 150000L;
	final double BASE_TIME_FRACTION = .08; // rough guess
	final int MIN_DEPTH_LIMIT = 2;
	final int MAX_DEPTH_LIMIT = 100;

	Stopwatch stopwatch = new Stopwatch();
	stopwatch.start();

	// We want to have a rough idea of how far we are through
	// the game both in terms of time and pieces.
	
	double playLeft = (double) piecesRemaining(node) / TOTAL_PIECES;
	double timeLeft = (double) timeRemaining / TOTAL_TIME;
	
	// We want to have a measure of whether we're getting
	// ahead or behind in our use of time.
	double adjustmentFactor = timeLeft / playLeft;

	// Finally, we use this measure with an approximate
	// measure of time per turn to compute a time limit for
	// this decision.
	double decisionTime 
	    = adjustmentFactor * TOTAL_TIME * BASE_TIME_FRACTION;
	
	// Create a new copy of the input node in my own node
	// type (with my own evaluation function)
	MBBWMancalaNode searchNode = new MBBWMancalaNode(node);

	// Perform successively deeper searches until we believe
	// we'll exceed our decision time in the next iteration.
	boolean done = false;
	int depthLimit = MIN_DEPTH_LIMIT;
	int bestMove = GameNode.UNDEFINED_MOVE;
	long timeTaken;
	while (!done && depthLimit <= MAX_DEPTH_LIMIT) {
	    MBBWAlphaBetaSearcher searcher = new MBBWAlphaBetaSearcher(depthLimit);
	    searcher.eval(searchNode);
	    bestMove = searcher.getBestMove();
	    timeTaken = stopwatch.lap();
	    // Since the branching factor is <= 6, we conservatively
	    // estimate that the next search will take 6 times as long
	    // as the searches that have already been done.  If the
	    // next search is thus estimated to exceed our allotted
	    // search time, we terminate iterative-deepening.
	    if (decisionTime / timeTaken < 3)
		done = true;
	    else
		depthLimit++;
	}
	System.out.println("Decision time: " + (int) decisionTime);
	System.out.println("Used time: " + stopwatch.lap());
	System.out.println("Depth Limit: " + depthLimit);
	return bestMove;	
    }

}
