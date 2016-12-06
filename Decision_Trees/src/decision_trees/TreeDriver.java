package decision_trees;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import weka.core.Instances;

public class TreeDriver {

    public final static String CANCER_TRAINING_DATA_FILE_PATH = "cancer_train.txt";
    public final static String CANCER_TESTING_DATA_FILE_PATH = "cancer_test.txt";
    public final static String MUSHROOM_TRAINING_DATA_FILE_PATH = "mushrooms_training.txt";
    public final static String MUSHROOM_TESTING_DATA_FILE_PATH = "mushroom_test.txt";

	public static final Instances LoadInstancesFromFile(String i_FileName) {
		try
		{
			BufferedReader inputReader = new BufferedReader(new FileReader(i_FileName));
			return new Instances(inputReader);
		}
		catch (FileNotFoundException ex)
		{
			System.err.println("File not found: " + i_FileName);
		}
		catch (IOException e)
		{
			System.out.println("IO error : " + e.toString());
		}

		return null;
	}

	public static void main(String[] args) throws Exception
	{
		DecisionTree decisionTree = new DecisionTree();

        // First run - learn and build the tree without pruning (CANCER)
        decisionTree.SetPruningMode(false);
		decisionTree.buildClassifier(LoadInstancesFromFile(CANCER_TRAINING_DATA_FILE_PATH));

        System.out.println("The percentage error on training data (CANCER) without pruning: "
				+  decisionTree.CalcAvgError(LoadInstancesFromFile(CANCER_TRAINING_DATA_FILE_PATH)));
        System.out.println("The percentage error on testing data (CANCER) without pruning: "
                +  decisionTree.CalcAvgError(LoadInstancesFromFile(CANCER_TESTING_DATA_FILE_PATH)));

        // Second run - learn and build the tree and then do pruning (CANCER)
        decisionTree.SetPruningMode(true);
        decisionTree.buildClassifier(LoadInstancesFromFile(CANCER_TRAINING_DATA_FILE_PATH));

        System.out.println("The percentage error on training (CANCER) data with pruning: "
                +  decisionTree.CalcAvgError(LoadInstancesFromFile(CANCER_TRAINING_DATA_FILE_PATH)));
        System.out.println("The percentage error on testing data (CANCER) with pruning: "
                +  decisionTree.CalcAvgError(LoadInstancesFromFile(CANCER_TESTING_DATA_FILE_PATH)));




        // First run - learn and build the tree without pruning (MUSHROOM)
        decisionTree.SetPruningMode(false);
        decisionTree.buildClassifier(LoadInstancesFromFile(MUSHROOM_TRAINING_DATA_FILE_PATH));

        System.out.println("The percentage error on training data (MUSHROOM) without pruning: "
                +  decisionTree.CalcAvgError(LoadInstancesFromFile(MUSHROOM_TRAINING_DATA_FILE_PATH)));
        System.out.println("The percentage error on testing data (MUSHROOM) without pruning: "
                +  decisionTree.CalcAvgError(LoadInstancesFromFile(MUSHROOM_TESTING_DATA_FILE_PATH)));

        // Second run - learn and build the tree and then do pruning (MUSHROOM)
        decisionTree.SetPruningMode(true);
        decisionTree.buildClassifier(LoadInstancesFromFile(MUSHROOM_TRAINING_DATA_FILE_PATH));

        System.out.println("The percentage error on training data (MUSHROOM) with pruning: "
                +  decisionTree.CalcAvgError(LoadInstancesFromFile(MUSHROOM_TRAINING_DATA_FILE_PATH)));
        System.out.println("The percentage error on testing data (MUSHROOM) with pruning: "
                +  decisionTree.CalcAvgError(LoadInstancesFromFile(MUSHROOM_TESTING_DATA_FILE_PATH)));

	}
}
