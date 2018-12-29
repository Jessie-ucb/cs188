# QuickSort implementation using list
# Using the first element as the pivot
def quickSort(inList):
	if len(inList) <= 1:
		return inList
		
	pivot = inList[0]
	smallList = [x for x in inList if x < pivot]
	equalList = [x for x in inList if x == pivot]
	largeList = [x for x in inList if x > pivot]
	orderedSmallList = quickSort(smallList)
	orderedLargeList = quickSort(largeList)
	return orderedSmallList + equalList + orderedLargeList

# main function for testing
if __name__ == '__main__':
	testList = [5, 2, 7, 1, 8, 9, 2, 3, 5, 6]
	orderedTestList = quickSort(testList)
	print(orderedTestList)