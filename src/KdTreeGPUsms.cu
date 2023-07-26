//
//  KdTreeGPUsms.cu
//
//  Created by John Robinson on 7/15/15.
//  Copyright (c) 2015 John Robinson. All rights reserved.
/*
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSEARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */
//

/*
 * Copyright (c) 2015, Russell A. Brown
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSEARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* @(#)kdTreeSingleThread.cc	1.61 04/13/15 */

/*
 * The k-d tree was described by Jon Bentley in "Multidimensional Binary Search Trees
 * Used for Associative Searching", CACM 18(9): 509-517, 1975.  For k dimensions and
 * n elements of data, a balanced k-d tree is built in O(kn log n) + O((k+1)n log n)
 * time by first sorting the data in each of k dimensions, then building the k-d tree
 * in a manner that preserves the order of the k sorts while recursively partitioning
 * the data at each level of the k-d tree. No further sorting is necessary.  Moreover,
 * it is possible to replace the O((k+1)n log n) term with a O((k-1)n log n) term but
 * this approach sacrifices the generality of building the k-d tree for points of any
 * number of dimensions.
 */

#include <stdbool.h>
#include <stdlib.h>
#include <vector>
#include <list>
#include <math.h>
#include <iostream>
#include <iomanip>
using std::setprecision;

using namespace std;

#include "Gpu.h"
#include "KdNode.h"

Gpu *gpu;

/*
 * The createKdTree function performs the necessary initialization then calls the buildKdTree function.
 *
 * calling parameters:
 *
 * coordinates - a vector<int*> of references to each of the (x, y, z, w...) tuples
 * numDimensions - the number of dimensions
 *
 * returns: a KdNode pointer to the root of the k-d tree
 */
KdNode* KdNode::createKdTree(KdNode kdNodes[], KdCoord coordinates[],  const sint numDimensions, const sint numTuples)
{

	Gpu::initializeKdNodesArray(coordinates, numTuples, numDimensions);
	cudaDeviceSynchronize();

	// Sort the reference array using multiple threads if possible.

	sint end[numDimensions]; // Array used to collect results of the remove duplicates function
	Gpu::mergeSort(end, numTuples, numDimensions);

	// Check that the same number of references was removed from each reference array.
	for (sint i = 0; i < numDimensions-1; i++) {
		if (end[i] < 0) {
			cout << "removeDuplicates failed on dimension " << i << endl;
			cout << end[0];
			for (sint k = 1;  k<numDimensions; k++) cout << ", " << end[k] ;
			cout << endl;
			exit(1);
		}
		for (sint j = i + 1; j < numDimensions; j++) {
			if ( end[i] != end[j] ) {
				cout << "Duplicate removal error" << endl;
				cout << end[0];
				for (sint k = 1;  k<numDimensions; k++) cout << ", " << end[k] ;
				cout << endl;
				exit(1);
			}
		}
	}
	cout << numTuples-end[0] << " equal nodes removed. "<< endl;

	// Build the k-d tree.
	//  refIdx_t root = gpu->startBuildKdTree(kdNodes, end[0], numDimensions);
	refIdx_t root = Gpu::buildKdTree(kdNodes, end[0], numDimensions);

	// Return the pointer to the root of the k-d tree.
	return &kdNodes[root];
}

/*
 * Search the k-d tree and find the KdNodes that lie within a cutoff distance
 * from a query node in all k dimensions.
 *
 * calling parameters:
 *
 * query - the query point
 * cut - the cutoff distance
 * dim - the number of dimensions
 * depth - the depth in the k-d tree
 *
 * returns: a list that contains the kdNodes that lie within the cutoff distance of the query node
 */
list<KdNode> KdNode::searchKdTree(const KdNode kdNodes[], const KdCoord coords[], const KdCoord* query, const KdCoord cut,
		const sint dim, const sint depth) const {

	// The partition cycles as x, y, z, w...
	sint axis = depth % dim;

	// If the distance from the query node to the k-d node is within the cutoff distance
	// in all k dimensions, add the k-d node to a list.
	list<KdNode> result;
	bool inside = true;
	for (sint i = 0; i < dim; i++) {
		if (abs(query[i] - coords[tuple*dim+i]) > cut) {
			inside = false;
			break;
		}
	}
	if (inside) {
		result.push_back(*this); // The push_back function expects a KdNode for a call by reference.
	}

	// Search the < branch of the k-d tree if the partition coordinate of the query point minus
	// the cutoff distance is <= the partition coordinate of the k-d node.  The < branch must be
	// searched when the cutoff distance equals the partition coordinate because the super key
	// may assign a point to either branch of the tree if the sorting or partition coordinate,
	// which forms the most significant portion of the super key, shows equality.
	if ( ltChild != -1 && (query[axis] - cut) <= coords[tuple*dim+axis] ) {
		list<KdNode> ltResult = kdNodes[ltChild].searchKdTree(kdNodes, coords, query, cut, dim, depth + 1);
		result.splice(result.end(), ltResult); // Can't substitute searchKdTree(...) for ltResult.
	}

	// Search the > branch of the k-d tree if the partition coordinate of the query point plus
	// the cutoff distance is >= the partition coordinate of the k-d node.  The < branch must be
	// searched when the cutoff distance equals the partition coordinate because the super key
	// may assign a point to either branch of the tree if the sorting or partition coordinate,
	// which forms the most significant portion of the super key, shows equality.
	if ( gtChild != -1 && (query[axis] + cut) >= coords[tuple*dim+axis] ) {
		list<KdNode> gtResult = kdNodes[gtChild].searchKdTree(kdNodes, coords, query, cut, dim, depth + 1);
		result.splice(result.end(), gtResult); // Can't substitute searchKdTree(...) for gtResult.
	}

	return result;
}

/*
 * Print one tuple.
 *
 * calling parameters:
 *
 * tuple - the tuple to print
 * dim - the number of dimensions
 */
void KdNode::printTuple(const KdCoord* tuple, const sint dim)
{
	cout << "(" << tuple[dim] << ",";
	for (sint i=1; i<dim-1; i++) cout << tuple[i] << ",";
	cout << tuple[dim-1] << ")";
}

/* Create a simple k-d tree and print its topology for inspection. */
sint main(sint argc, char **argv)
{
	// Set the defaults then parse the input arguments.
	sint numPoints = 4194304;
	sint extraPoints = 100;
	sint numDimensions = 3;
	sint numThreads = 512;
	sint numBlocks = 32;
	sint searchDistance = 20000000;
	sint maximumNumberOfNodesToPrint = 5;

	for (sint i = 1; i < argc; i++) {
		if ( 0 == strcmp(argv[i], "-n") || 0 == strcmp(argv[i], "--numPoints") ) {
			numPoints = atol(argv[++i]);
			continue;
		}
		if ( 0 == strcmp(argv[i], "-x") || 0 == strcmp(argv[i], "--extraPoints") ) {
			extraPoints = atol(argv[++i]);
			continue;
		}
		if ( 0 == strcmp(argv[i], "-d") || 0 == strcmp(argv[i], "--numDimensions") ) {
			numDimensions = atol(argv[++i]);
			continue;
		}
		if ( 0 == strcmp(argv[i], "-t") || 0 == strcmp(argv[i], "--numThreads") ) {
			numThreads = atol(argv[++i]);
			continue;
		}
		if ( 0 == strcmp(argv[i], "-b") || 0 == strcmp(argv[i], "--numBlocks") ) {
			numBlocks = atol(argv[++i]);
			continue;
		}
		if ( 0 == strcmp(argv[i], "-s") || 0 == strcmp(argv[i], "--searchDistance") ) {
			searchDistance = atol(argv[++i]);
			continue;
		}
		if ( 0 == strcmp(argv[i], "-p") || 0 == strcmp(argv[i], "--maximumNodesToPrint") ) {
			maximumNumberOfNodesToPrint = atol(argv[++i]);
			continue;
		}
		cout << "Unsupported command-line argument: " <<  argv[i] << endl;
		exit(1);
	}

	sint  i = maximumNumberOfNodesToPrint + numDimensions + extraPoints;
	// Declare the two-dimensional coordinates array that contains (x,y,z) coordinates.
	/*
    sint coordinates[NUM_TUPLES][DIMENSIONS] = {
    {2,3,3}, {5,4,2}, {9,6,7}, {4,7,9}, {8,1,5},
    {7,2,6}, {9,4,1}, {8,4,2}, {9,7,8}, {6,3,1},
    {3,4,5}, {1,6,8}, {9,5,3}, {2,1,3}, {8,7,6},
    {5,4,2}, {6,3,1}, {8,7,6}, {9,6,7}, {2,1,3},
    {7,2,6}, {4,7,9}, {1,6,8}, {3,4,5}, {9,4,1} };
	 */
	//  gpu = new Gpu(numThreads,numBlocks,0,numDimensions);
	Gpu::gpuSetup(2, numThreads,numBlocks,numDimensions);
	if (Gpu::getNumThreads() == 0 || Gpu::getNumBlocks() == 0) {
		cout << "KdNode Tree cannot be built with " << numThreads << " threads or " << numBlocks << " blocks." << endl;
		exit(1);
	}
	cout << "Points = " << numPoints << " dimensions = " << numDimensions << ", threads = " << numThreads << ", blocks = " << numBlocks << endl;

	srand(0);
	KdCoord (*coordinates) = new KdCoord[numPoints*numDimensions];
	for ( i = 0; i<numPoints; i++) {
		for (sint j=0; j<numDimensions; j++) {
			coordinates[i*numDimensions+j] = (KdCoord)rand();
			//coordinates[i*numDimensions+j] = (j==1)? (numPoints-i) : i;
			//coordinates[i*numDimensions+j] =  i;
		}
	}

	// Create the k-d tree.  First copy the data to a tuple in its kdNode.
	// also null out the gt and lt references
	// create and initialize the kdNodes array
	KdNode *kdNodes = new KdNode[numPoints];
	if (kdNodes == NULL) {
		printf("Can't allocate %d kdNodes\n", numPoints);
		exit (1);
	}

	KdNode *root = KdNode::createKdTree(kdNodes, coordinates, numDimensions, numPoints);

	// Print the k-d tree "sideways" with the root at the left.
	cout << endl;

	if (searchDistance == 0){
		return 0;
	}
	// Search the k-d tree for the k-d nodes that lie within the cutoff distance of the first tuple.
	KdCoord* query = (KdCoord *)malloc(numDimensions * sizeof(KdCoord));
	for (sint i = 0; i < numDimensions; i++) {
		query[i] = coordinates[i];
	}
	// read the KdTree back from GPU
	Gpu::getKdTreeResults( kdNodes,  coordinates, numPoints, numDimensions);
	list<KdNode> kdList = root->searchKdTree(kdNodes, coordinates, query, searchDistance, numDimensions, 0);

	cout << endl << kdList.size() << " nodes within " << searchDistance << " units of ";
	KdNode::printTuple(query, numDimensions);
	cout << " in all dimensions." << endl << endl;
	if (kdList.size() != 0) {
		cout << "List of k-d nodes within " << searchDistance << "-unit search distance follows:" << endl << endl;
		list<KdNode>::iterator it;
		for (it = kdList.begin(); it != kdList.end(); it++) {
			KdNode::printTuple(coordinates+it->getTuple()*numDimensions, numDimensions);
			cout << " ";
		}
		cout << endl;
	}
	return 0;
}
