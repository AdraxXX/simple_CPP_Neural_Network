/*
 * Created By: Daniel Johnston
 * 
 * Project: Simple neural network with the option to create a network with any size of hidden layers, input nodes and output nodes.
 * 
 * Data: 06 November 2019
 * 
 * Uploaded: 07 November 2019
 */
#include <math.h>
#include <time.h>
#include <iostream>
#include <string>
#include <vector>

using namespace std;


struct NeuronLayer;
struct Neuron;


//layout for the neuron in neural network
struct Neuron
{
	bool isInputLayer;
	bool isOutputLayer;
	float data;
	float bias;
	string informationStored;
	vector<float> weight;
};


//layout for the layer of neurons in neural network
struct NeuronLayer
{
	int sizeOfLayer;
	vector<Neuron> Nodes;
};


//This class is the layout for an abstract neural network
class NeuralNetwork
{
	private:
		int numberOfHiddenLayers;
		float learningRateOfNeuralNetwork;
		NeuronLayer inputNodes;
		NeuronLayer outputNodes;
		vector<NeuronLayer> hiddenNodes;
		float sigmoidFunction(float);
		vector<float> processingInformation(NeuronLayer, vector<float>, int);
		void Backpropagation(NeuronLayer&, float, int, int);
		void changeBiasOrWeight(float, float&);
	public:
		NeuralNetwork(int, int, int, vector<string>, float);
		void thoughtProcessing(vector<float>, int);
		void trainingProcess(vector<float>, vector<float>, int, int);
		void printNeuralNetwork();
};


// This function changes a current weight or bias within the neural network
void NeuralNetwork::changeBiasOrWeight(float delta, float& biasOrWeight)
{
	biasOrWeight = biasOrWeight - (learningRateOfNeuralNetwork * delta);
}


// This function goes back through the neural network and tries to modify weights for better results
void NeuralNetwork::Backpropagation(NeuronLayer& currentLayer, float error, int currentHiddenLayer, int indexLastNode)
{
	float sumOfWeights = 0.0;
	// This for loop runs through getting the sum of all of the weights used to calculate the last node
	for(int indexCurrentLayer = 0; indexCurrentLayer < currentLayer.sizeOfLayer; indexCurrentLayer++)
	{
		if(currentHiddenLayer >= 0)
		{
			sumOfWeights += currentLayer.Nodes.at(indexCurrentLayer).weight.at(indexLastNode) * currentLayer.Nodes.at(indexCurrentLayer).bias;
		}
		else
		{
			sumOfWeights += currentLayer.Nodes.at(indexCurrentLayer).weight.at(indexLastNode) * currentLayer.Nodes.at(indexCurrentLayer).data;
		}
	}
	// This for loop call the backpropagration on all of the weights behind the current layer and changes the current weights and bias
	// information of the current layer.
	for(int indexCurrentLayer = 0; indexCurrentLayer < currentLayer.sizeOfLayer; indexCurrentLayer++)
	{
		float biasDelta = (-2.0) * (error) * sigmoidFunction(sumOfWeights) * ( 1.0 - sigmoidFunction(sumOfWeights) );
		float delta = biasDelta * currentLayer.Nodes.at(indexCurrentLayer).data;
		if(currentHiddenLayer >= 0)
		{
			float delta = biasDelta * currentLayer.Nodes.at(indexCurrentLayer).bias;
			if(currentHiddenLayer != 0)
			{
				Backpropagation(hiddenNodes.at(currentHiddenLayer - 1), delta, currentHiddenLayer - 1, indexCurrentLayer);
			}
			else
			{
				Backpropagation(inputNodes, delta, currentHiddenLayer - 1, indexCurrentLayer);
			}
			changeBiasOrWeight(biasDelta, currentLayer.Nodes.at(indexCurrentLayer).bias);
		}
		changeBiasOrWeight(delta, currentLayer.Nodes.at(indexCurrentLayer).weight.at(indexLastNode));
	}
}


// this function allows the user to train the neural network
void NeuralNetwork::trainingProcess(vector<float> data, vector<float> outcomes, int amountOfData, int numberOfRounds)
{
	// This for loop runs through the training set as many times as the user input
	for (int indexNumberOfRounds = 0; indexNumberOfRounds < numberOfRounds; indexNumberOfRounds++)
	{
		// This for loop runs through the entire set of input data
		for(int indexDataSet = 0; indexDataSet < amountOfData ; indexDataSet += inputNodes.sizeOfLayer)
		{
			// This for loop stores all of the input data into the input nodes
			for(int indexData = 0; indexData < inputNodes.sizeOfLayer; indexData++)
			{
				inputNodes.Nodes.at(indexData).data = data.at(indexDataSet + indexData);
			}
			// processes all of the information
			vector<float> output = processingInformation(inputNodes, {0.0}, 0);
			// This for loop runs through the backprogration of the output nodes
			for(int indexOutputNodes = 0; indexOutputNodes < outputNodes.sizeOfLayer; indexOutputNodes++)
			{
				Backpropagation(hiddenNodes.at(numberOfHiddenLayers - 1), (output.at(indexOutputNodes) - outcomes.at(indexOutputNodes + indexDataSet))
								, numberOfHiddenLayers - 1, indexOutputNodes);
				changeBiasOrWeight((output.at(indexOutputNodes) - outcomes.at(indexOutputNodes + indexDataSet)), outputNodes.Nodes.at(indexOutputNodes).bias);
			}
		}
	}
}


// This allows the user to input data to be processed by the neural network
void NeuralNetwork::thoughtProcessing(vector<float> data, int amountOfData)
{
	// This for loop runs through the entire set of input data returning the results of the inputs
	for(int indexDataSet = 0; indexDataSet < amountOfData * inputNodes.sizeOfLayer ; indexDataSet += inputNodes.sizeOfLayer)
	{
		cout << "Results from input data: ";
		// This for loop stores all of the input data into the input nodes
		for(int indexData = 0; indexData < inputNodes.sizeOfLayer; indexData++)
		{
			cout << data.at(indexDataSet + indexData);
			if( indexData != inputNodes.sizeOfLayer - 1)
			{
				cout << ", ";
			}
			inputNodes.Nodes.at(indexData).data = data.at(indexDataSet + indexData);
		}
		cout << endl;
		vector<float> output = processingInformation(inputNodes, {0.0}, 0);
		// This for loop outputs all of the results to the inputs
		for(int indexOutputNodes = 0; indexOutputNodes < outputNodes.sizeOfLayer; indexOutputNodes++)
		{
			cout << "Current Output Node: " << indexOutputNodes << " " << output.at(indexOutputNodes) << endl;
		}
		cout << endl;
	}
}


// This function is called to process the current data through the neural network
vector<float> NeuralNetwork::processingInformation(NeuronLayer currentLayer, vector<float> currentData, int hiddenLayerIndex)
{
	if(currentLayer.Nodes.at(0).isInputLayer == true)
	{
		currentData.clear();
		// This for loop sets all of the data to 0.0 to start
		for(int indexWeight = 0; indexWeight < hiddenNodes.at(0).sizeOfLayer; indexWeight++)
		{
			currentData.push_back(0.0);
		}
		// This for loop loops through the current layer getting the new set of data;
		for(int indexLayer = 0; indexLayer < currentLayer.sizeOfLayer; indexLayer++)
		{
			for(int indexWeight = 0; indexWeight < hiddenNodes.at(0).sizeOfLayer; indexWeight++)
			{
				currentData.at(indexWeight) += currentLayer.Nodes.at(indexLayer).data * currentLayer.Nodes.at(indexLayer).weight.at(indexWeight);
			}
		}
		return processingInformation(hiddenNodes.at(0), currentData, 0);
	}
	else if(currentLayer.Nodes.at(0).isOutputLayer == true)
	{
		// This for loop sets uses the sigmoid function on all of the (weight * data) + bias to make our final data
		for(int indexCurrentLayer = 0; indexCurrentLayer < currentLayer.sizeOfLayer; indexCurrentLayer++)
		{
			currentData.at(indexCurrentLayer) = sigmoidFunction(currentData.at(indexCurrentLayer) + currentLayer.Nodes.at(indexCurrentLayer).bias);
		}
		return currentData;
	}
	else
	{
		vector<float> newData;
		// This for loop sets uses the sigmoid function on all of the (weight * data) + bias to make our new data at this node
		for(int indexCurrentLayer = 0; indexCurrentLayer < currentLayer.sizeOfLayer; indexCurrentLayer++)
		{
			currentData.at(indexCurrentLayer) = sigmoidFunction(currentData.at(indexCurrentLayer) + currentLayer.Nodes.at(indexCurrentLayer).bias);
		}
		if(hiddenLayerIndex == numberOfHiddenLayers - 1)
		{
			// This for loop sets all of the data to 0.0 to start
			for(int indexNextLayer = 0; indexNextLayer < outputNodes.sizeOfLayer;indexNextLayer++)
			{
				newData.push_back(0.0);
			}
		}
		else
		{
			// This for loop sets all of the data to 0.0 to start
			for(int indexNextLayer = 0; indexNextLayer < hiddenNodes.at(hiddenLayerIndex + 1).sizeOfLayer;indexNextLayer++)
			{
				newData.push_back(0.0);
			}
		}
		
		// This for loop loops through the current layer getting the new set of data;
		for(int indexCurrentLayer = 0; indexCurrentLayer < currentLayer.sizeOfLayer; indexCurrentLayer++)
		{
			if(hiddenLayerIndex == numberOfHiddenLayers - 1)
			{
				// This for loop loops through the weights to make (weight * data) for the next node layer
				for(int indexWeight = 0; indexWeight < outputNodes.sizeOfLayer; indexWeight++)
				{
					newData.at(indexWeight) +=  currentData.at(indexCurrentLayer) * currentLayer.Nodes.at(indexCurrentLayer).weight.at(indexWeight);
				}
			}
			else
			{
				// This for loop loops through the weights to make (weight * data) for the next node layer
				for(int indexWeight = 0; indexWeight < hiddenNodes.at(hiddenLayerIndex + 1).sizeOfLayer; indexWeight++)
				{
					newData.at(indexWeight) +=  currentData.at(indexCurrentLayer) * currentLayer.Nodes.at(indexCurrentLayer).weight.at(indexWeight);
				}
			}
		}
		if(hiddenLayerIndex == numberOfHiddenLayers - 1)
		{
			return processingInformation(outputNodes, newData, hiddenLayerIndex);
		}
		else
		{
			return processingInformation(hiddenNodes.at(hiddenLayerIndex + 1), newData, hiddenLayerIndex + 1);
		}
	}
}


// This function allows the program to use the sigmoid function
float NeuralNetwork::sigmoidFunction(float number)
{
	return (1.0 / (1.0 + exp(-number)));
}


// Parameterized Constructor 
NeuralNetwork::NeuralNetwork(int numberOfInputNodes, int numberOfOutputNodes, int numberOfHiddenLayers, vector<string> outputItems, float learningRateOfNeuralNetwork)
{
	srand (time(NULL));
	NeuronLayer newInputLayer;
	NeuronLayer newOutputNodes;
	this->learningRateOfNeuralNetwork = learningRateOfNeuralNetwork;
	this->numberOfHiddenLayers = numberOfHiddenLayers;
	
	newOutputNodes.sizeOfLayer = numberOfOutputNodes;
	newInputLayer.sizeOfLayer = numberOfInputNodes;
	
	// This for loop loops through all of the hidden layers creating them based on users input information
	for(int indexHiddenLayers = 0; indexHiddenLayers < numberOfHiddenLayers; indexHiddenLayers++)
	{
		NeuronLayer newHiddenLayer;
		if(indexHiddenLayers == numberOfHiddenLayers - 1)
		{
			if(numberOfHiddenLayers != 1)
			{
				newHiddenLayer.sizeOfLayer = (2.0/3.0) * numberOfInputNodes + hiddenNodes.at(0).sizeOfLayer;
			}
			else
			{
				newHiddenLayer.sizeOfLayer = (2.0/3.0) * numberOfInputNodes + numberOfOutputNodes;
			}
		}
		else if(indexHiddenLayers == 0)
		{
			newHiddenLayer.sizeOfLayer = (2.0/3.0) * numberOfInputNodes + numberOfOutputNodes;
		}
		else
		{
			newHiddenLayer.sizeOfLayer = (2.0/3.0) * hiddenNodes.at(0).sizeOfLayer + hiddenNodes.at(hiddenNodes.size() -1).sizeOfLayer;
		}
		
		// This for loop is used to create new neurons within the given hidden layer
		for(int indexHiddenNodes = 0; indexHiddenNodes < newHiddenLayer.sizeOfLayer; indexHiddenNodes++)
		{
			Neuron newNeuron;
			newNeuron.isInputLayer      = false;
			newNeuron.isOutputLayer     = false;
			newNeuron.data              = 0.0;
			newNeuron.bias              = 0.0;
			newNeuron.informationStored = "";
			if(indexHiddenLayers == 0)
			{
				// This for loop goes through the the last hidden node layer assigning weights corresponding to the nodes within the output node layer
				for(int indexOutputNodes = 0; indexOutputNodes < numberOfOutputNodes; indexOutputNodes++)
				{
					newNeuron.weight.push_back(sin((float)(rand())));
				}
			}
			else
			{
				// This for loop goes through the current hidden layer of nodes assigning weights coresponding to the last hidden layer of nodes that was made.
				for(int indexHiddenNodesOfLastLayer = 0; indexHiddenNodesOfLastLayer < hiddenNodes.at(0).sizeOfLayer; indexHiddenNodesOfLastLayer++)
				{
					newNeuron.weight.push_back(sin((float)(rand())));
				}
			}
			newHiddenLayer.Nodes.push_back(newNeuron);
		}
		hiddenNodes.insert(hiddenNodes.begin(), newHiddenLayer);
	}
	
	// This for loop that goes through the input layer creating each node based on the users input.
	for(int indexInputLayer = 0; indexInputLayer < numberOfInputNodes; indexInputLayer++)
	{
		Neuron newNeuron;
		newNeuron.isInputLayer      = true;
		newNeuron.isOutputLayer     = false;
		newNeuron.data              = 0.0;
		newNeuron.bias              = 0.0;
		newNeuron.informationStored = "";
		// This for loop goes through the first HiddenNodes assigning weights corresponding to the nodes within the hidden layer
		for(int indexFirstHiddenNodes = 0; indexFirstHiddenNodes < hiddenNodes.at(0).sizeOfLayer; indexFirstHiddenNodes++)
		{
			newNeuron.weight.push_back(sin((float)(rand())));
		}
		newInputLayer.Nodes.push_back(newNeuron);
	}
	inputNodes = newInputLayer;
	
	// This for loop that goes through the output layer creating each node based on the users input.
	for(int indexOutputLayer = 0; indexOutputLayer < numberOfOutputNodes; indexOutputLayer++)
	{
		Neuron newNeuron;
		newNeuron.isInputLayer      = false;
		newNeuron.isOutputLayer     = true;
		newNeuron.data              = 0.0;
		newNeuron.bias              = 0.0;
		newNeuron.informationStored = outputItems.at(indexOutputLayer);
		newOutputNodes.Nodes.push_back(newNeuron);
	}
	outputNodes = newOutputNodes;
}


// Function used to show the current neural network system information
void NeuralNetwork::printNeuralNetwork()
{
	cout << "Input Layers" << endl;
	// This for loop is used to output the current input Nodes in the neural network
	for(int indexInputNodes = 0; indexInputNodes < inputNodes.sizeOfLayer; indexInputNodes++)
	{
		cout << "Current Input Neuron: " << indexInputNodes << endl;
		cout << "Size of weight vector: " << inputNodes.Nodes.at(indexInputNodes).weight.size() << endl;
		for(int indexWeights = 0; indexWeights < hiddenNodes.at(0).sizeOfLayer; indexWeights++)
		{
			cout << "     Current Weight: " << indexWeights << " " << inputNodes.Nodes.at(indexInputNodes).weight.at(indexWeights) << endl;
		}
	}
	
	cout << endl << endl << "Hidden Layers" << endl;
	// This for loop is used to output all of the hidden layers of nodes
	for(int indexHiddenLayer = 0; indexHiddenLayer < numberOfHiddenLayers; indexHiddenLayer++)
	{
		cout << "Current Hidden Layer: " << indexHiddenLayer << endl;
		// This for loop is used to move through all of the hidden nodes of a layer
		for(int indexHiddenNodes = 0; indexHiddenNodes < hiddenNodes.at(indexHiddenLayer).sizeOfLayer; indexHiddenNodes++)
		{
			cout << "     Current Hidden Neuron: " << indexHiddenNodes << endl;
			cout << "     Current Hidden Neuron Bias: " << hiddenNodes.at(indexHiddenLayer).Nodes.at(indexHiddenNodes).bias << endl;
			cout << "     Size of weight vector: " << hiddenNodes.at(indexHiddenLayer).Nodes.at(indexHiddenNodes).weight.size() << endl;
			if(indexHiddenLayer != numberOfHiddenLayers - 1)
			{
				// This for loop moves through all of the weights used to move to the next layer
				for(int indexWeights = 0; indexWeights < hiddenNodes.at(indexHiddenLayer + 1).sizeOfLayer; indexWeights++)
				{
					cout << "          Current Weight: " << indexWeights << " " << hiddenNodes.at(indexHiddenLayer).Nodes.at(indexHiddenNodes).weight.at(indexWeights) << endl;
				}
			}
			else
			{
				// This for loop moves through all of the weights used to move to the next layer
				for(int indexWeights = 0; indexWeights < outputNodes.sizeOfLayer; indexWeights++)
				{
					cout << "          Current Weight: " << indexWeights << " " << hiddenNodes.at(indexHiddenLayer).Nodes.at(indexHiddenNodes).weight.at(indexWeights) << endl;
				}
			}
		}
		
		
	}
	
	cout << endl << endl << "Output Layer" << endl;
	// This for loop is used to output the output node layer
	for(int indexOutputNodes = 0; indexOutputNodes < outputNodes.sizeOfLayer; indexOutputNodes++)
	{
		cout << "Current Output Neuron: " << indexOutputNodes << endl;
		cout << "Current Output Neuron Bias: " << outputNodes.Nodes.at(indexOutputNodes).bias << endl;
		cout << "Current Output Neuron Value: " << outputNodes.Nodes.at(indexOutputNodes).informationStored << endl;
	}
}
