package decision_trees;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import java.util.ArrayList;
import java.util.List;

public class DecisionTree  {

	private final String PRUNING_ROOT_NULL_ERROR_MSG = "Tree root is null. incapable to perform pruning.";

	private final double CHI_SQUARE_COMPARE_NUMBER = 2.733;

	private int m_NumOfAttributes;
	private boolean m_PruningMode = false;
	private TreeNode m_Root;

	public void buildClassifier(Instances i_Instances) throws Exception
	{
		i_Instances.setClassIndex(i_Instances.numAttributes() - 1);

		m_NumOfAttributes = i_Instances.numAttributes() - 1;

		// Build list of instance
		List<Instance> instancesList = new ArrayList<>();
		for(int i = 0; i < i_Instances.numInstances(); i++)
			instancesList.add(i_Instances.instance(i));

		// Build all the decision tree
		m_Root = buildTree(instancesList, null, -1.0);

		if(m_PruningMode) // Default is false
		{
			if(m_Root == null) {
				
				System.out.println(PRUNING_ROOT_NULL_ERROR_MSG);
			}
			else {

				Pruning(m_Root);

			}
		}

	}

	public void SetPruningMode(boolean i_PruningMode){
		m_PruningMode = i_PruningMode;
	}

	/*
	 * Computes the average error on spesific data set, 0 is the best, 1 is the worst
	 */
	public double CalcAvgError(Instances i_Instances)
	{
		int errorNumber = 0;

		i_Instances.setClassIndex(i_Instances.numAttributes() - 1);
		for(int i = 0; i < i_Instances.numInstances(); i++) {
			if(i_Instances.instance(i).classValue() != Classify(i_Instances.instance(i))) {
				errorNumber++;
			}
		}

		return ((double) errorNumber) / ((double) i_Instances.numInstances());
	}


	public double Classify(Instance i_Instance)
	{
		TreeNode currentNode = m_Root;

		while( ! currentNode.IsLeaf) {
			currentNode = currentNode.Children[ (int) i_Instance.value(currentNode.AttributeIndex)];
		}

		return currentNode.ClassValue;
	}


	private void Pruning(TreeNode i_TreeNode)
	{
		// Stopping condition
		if(i_TreeNode.IsLeaf) {
			return;
		}

		for(int i = 0; i < i_TreeNode.Children.length; i++) {
			Pruning(i_TreeNode.Children[i]);
		}

		if(calcChiSquare(i_TreeNode.Instances, i_TreeNode.AttributeIndex) > CHI_SQUARE_COMPARE_NUMBER) {
			// Means the split is ok
			return;

		} else {
			// Means the split is NOT significant, and we need to check if to  cut the branch
			TreeNode parent = i_TreeNode.Parent;

			// cutting of the branch.
			parent.Children[ (int) i_TreeNode.AttributeValueInParent] = new TreeNode(parent, i_TreeNode.AttributeValueInParent);
			parent.Children[ (int) i_TreeNode.AttributeValueInParent].IsLeaf = true;
			parent.Children[ (int) i_TreeNode.AttributeValueInParent].ClassValue = mostSignificantClassValue(i_TreeNode.Instances);

			// Release the reference to the parent
			i_TreeNode.Parent = null;

		}
	}


	private TreeNode buildTree(List<Instance> i_Instances, TreeNode i_Parent, double i_AttributeValueInParent)
	{
		TreeNode resultTree = new TreeNode(i_Parent, i_AttributeValueInParent);
		double bestGainValue = 0;
		int bestGainAttributeIndex;

		// Stopping conditions:

		// 1) all instances has the same class value
		if(isPureInstancesGroup(i_Instances)) {
			resultTree.IsLeaf = true;
			resultTree.ClassValue = i_Instances.isEmpty() ?
					mostSignificantClassValue(i_Parent.Instances)
					:
						i_Instances.get(0).classValue();
		}

		// 2) all the attributes in all the instances are the same
		else if(isDataEqual(i_Instances)) {
			resultTree.IsLeaf = true;
			resultTree.ClassValue = mostSignificantClassValue(i_Instances);
		}

		// Recursive step:
		else {
			// If here, the current iteration is not leaf.
			// Computes this node entropy (Target entropy)
			int positiveClassValue = 0;
			int negativeClassValue = 0;
			double thisNodeEntropy;
			for (int i = 0; i < i_Instances.size(); i++) {
				if (i_Instances.get(i).classValue() > 0)
					positiveClassValue++;
				else
					negativeClassValue++;
			}

			thisNodeEntropy = calcEntropy(positiveClassValue, negativeClassValue);
			bestGainAttributeIndex = 0;

			for (int attributeIndex = 0; attributeIndex < m_NumOfAttributes; attributeIndex++) {

				double currentAttributeGain = calcGain(i_Instances, thisNodeEntropy, attributeIndex);

				if (currentAttributeGain > bestGainValue) {
					bestGainValue = currentAttributeGain;
					bestGainAttributeIndex = attributeIndex;
				}
			}

			Attribute chosenAttribute = i_Instances.get(0).attribute(bestGainAttributeIndex);

			List<TreeNode> currentNodeChildren = new ArrayList<>();

			for (double attributeValue = 0.0; attributeValue < chosenAttribute.numValues(); attributeValue++) {

				List<Instance> childInstances = new ArrayList<>();

				for (int j = 0; j < i_Instances.size(); j++) {
					if (i_Instances.get(j).value(bestGainAttributeIndex) == attributeValue) {
						childInstances.add(i_Instances.get(j));
					}
				}

				currentNodeChildren.add(buildTree(childInstances, resultTree, attributeValue));
			}

			TreeNode[] currentNodeChildrenAsArray = currentNodeChildren.toArray(new TreeNode[currentNodeChildren.size()]);
			resultTree.Children = currentNodeChildrenAsArray;
			resultTree.IsLeaf = false;
			resultTree.AttributeIndex = bestGainAttributeIndex;
			resultTree.AttributeName = i_Instances.get(0).attribute(bestGainAttributeIndex).name();

			for(int i = 0; i < i_Instances.size(); i++) {
				resultTree.Instances.add(i_Instances.get(i));
			}
		}

		return resultTree;
	}

	/*
	 * computes the gain for specific attribute
	 * relative to before.
	 */
	private double calcGain(List<Instance> i_Instances, double i_TargetEntropy, int i_AttributeIndex)
	{
		double totalAttributeEntropy = 0;
		double numOfAttributeValues = i_Instances.get(0).attribute(i_AttributeIndex).numValues();
		int numOfInstances = i_Instances.size();

		for(int attributeValue = 0; attributeValue < numOfAttributeValues; attributeValue++) {

			int numOfPositivePerValue = 0;
			int numOfNegativePerValue = 0;

			for(int j = 0; j < i_Instances.size(); j++) {
				if(i_Instances.get(j).value(i_AttributeIndex) == attributeValue) {
					if(i_Instances.get(j).classValue() == 1)
						numOfPositivePerValue++;
					else
						numOfNegativePerValue++;
				}
			}

			if(numOfPositivePerValue != 0 && numOfNegativePerValue != 0) {
				totalAttributeEntropy += ((numOfNegativePerValue + numOfPositivePerValue) / (double) numOfInstances) *
						calcEntropy(numOfPositivePerValue, numOfNegativePerValue);
			}
		}

		return i_TargetEntropy - totalAttributeEntropy;
	}

	/*
	 * Check if group if instances is 'pure' group.
	 *  meaning they have the same class value
	 */
	private boolean isPureInstancesGroup(List<Instance> i_Instances)
	{
		if(i_Instances.isEmpty()) {
			return true;
		}

		boolean pureResult = true;
		double classValueToCheck = i_Instances.get(0).classValue();

		for(int i = 1; i < i_Instances.size(); i++)
		{
			if(i_Instances.get(i).classValue() != classValueToCheck) {
				pureResult = false;
				break;
			}
		}

		return pureResult;
	}

	/*
	 * check if all the attributes in all the instances are the same
	 */
	private boolean isDataEqual(List<Instance> i_Instances)
	{
		boolean result = true;
		int numAttributes = i_Instances.get(0).numAttributes() - 1;

		for(int i = 0; i < numAttributes; i++) {

			double currentAttributeValue = i_Instances.get(0).value(i);

			for(int j = 1; j < i_Instances.size(); j++)
			{
				if(i_Instances.get(j).value(i) != currentAttributeValue) {
					result = false;
					break;
				}
			}

			if(!result)
				break;
		}

		return result;
	}

	private double calcEntropy(int i_NumOfPositive, int i_NumOfNegative)
	{
		int totalGroupNumber = i_NumOfPositive + i_NumOfNegative;
		double positiveRatio = (double) i_NumOfPositive/ totalGroupNumber;
		double negativeRatio = (double) i_NumOfNegative/totalGroupNumber;

		return -(positiveRatio*log2(positiveRatio))-(negativeRatio*log2(negativeRatio));
	}

	private double mostSignificantClassValue(List<Instance> i_Instances) {
		int positive = 0, negative = 0;

		for(Instance instance : i_Instances) {
			if(instance.classValue() > 0.0)
				positive++;
			else
				negative++;
		}

		return positive > negative ? 1 : 0;
	}

	private double calcChiSquare(List<Instance> i_Instances, int i_AttributeIndex)
	{
		double result = 0;
		int numAttributeValues = i_Instances.get(0).attribute(i_AttributeIndex).numValues();
		int classValueIndex = i_Instances.get(0).numAttributes() - 1;

		// Variables for calculation
		int Df, Pf, Nf;
				double E0, E1;

				// Computes class values probability's
				double classValue0Probability = 0;
				double classValue1Probability = 0;
				for(int i = 0; i < i_Instances.size(); i++) {
					if(i_Instances.get(i).value(classValueIndex) == 0.0)
						classValue0Probability++;
					else
						classValue1Probability++;
				}

				classValue0Probability = classValue0Probability / (double) i_Instances.size();
				classValue1Probability = classValue1Probability / (double) i_Instances.size();

				for(double attributeValue = 0.0; attributeValue < numAttributeValues; attributeValue++) {
					// Initialize
					Df = 0;
					Pf = 0;
					Nf = 0;

					// Computes Df, Pf and Nf:
					for(int i = 0; i < i_Instances.size(); i++) {
						if(i_Instances.get(i).value(i_AttributeIndex) == attributeValue) {
							Df++;
							if(i_Instances.get(i).value(classValueIndex) == 0.0)
								Pf++;
							else
								Nf++;
						}
					}

					E0 = Df * classValue0Probability;
					E1 = Df * classValue1Probability;

					double leftPart = 0;
					double rightPart = 0;

					if(E0 != 0) {
						leftPart = (Math.pow(E0 - Pf, 2) / E0);
					}
					if(E1 != 0) {
						rightPart = (Math.pow(E1 - Nf, 2) / E1);
					}

					result += (leftPart + rightPart);
				}

				return result;
	}

	private double log2(double x) {
		return Math.log(x)/Math.log(2.0d);
	}

	private String innerToString(TreeNode i_TreeNode) {

		StringBuilder stringBuilder = new StringBuilder();

		if(i_TreeNode.IsLeaf) {
			stringBuilder.append("Leaf. parent name: " + i_TreeNode.Parent.AttributeName + ", Class value: "
					+ i_TreeNode.ClassValue + System.lineSeparator());
			return stringBuilder.toString();
		}

		for(int i = 0; i < i_TreeNode.Children.length; i++) {
			stringBuilder.append(innerToString(i_TreeNode.Children[i]));
		}

		stringBuilder.append("Node name: " + i_TreeNode.AttributeName + ", Node parent: " +
				(i_TreeNode.Parent != null ? i_TreeNode.Parent.AttributeName : "No parent, this is root") + System.lineSeparator() );

		return stringBuilder.toString();
	}

	@Override
	public String toString() {

		StringBuilder result = new StringBuilder();
		String[] lines = innerToString(m_Root).split(System.lineSeparator());

		for(int i = lines.length - 1; i >= 0; i--) {
			result.append(lines[i] + System.lineSeparator());
		}

		return result.toString();
	}


	/*
	 * Nested inner class that represent a single tree node.
	 */
	private class TreeNode
	{
		TreeNode Parent = null;
		TreeNode[] Children = null;
		int AttributeIndex = -1;
		String AttributeName = null;
		boolean IsLeaf = true;
		double ClassValue = -1;
		double AttributeValueInParent = -1;
		List<Instance> Instances;

		public TreeNode(TreeNode i_Parent, double i_AttributeValueInParent) {
			Parent = i_Parent;
			AttributeValueInParent = i_AttributeValueInParent;
			Instances = new ArrayList<>();
		}
	}
}
