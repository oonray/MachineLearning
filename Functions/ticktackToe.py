
row = 3
col = 3

array3 = {"Spam":4}
array2=[["Spam"]*row]*col
array=[[0]*row]*col

print(array)
print(array2)

for i in range(len(array)):
    for n in range(0,len(array[i])):
        array[i][n]= array3[array2[i][n]]


print(array)

col = [sum(i) for i in zip(*array)]
row = [sum(i) for i in array]
diag = [sum((array[0][0],array[1][1],array[2][2])),
        sum((array[0][2],array[1][1],array[2][0]))]

score = col+row+diag

if 12 in score:
    print("Player2 Wins")
elif 3 in score:
    print("Player 1 wins")
else:
    print("Tie")
