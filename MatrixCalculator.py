class MatrixSyntaxError(Exception):
    pass

class Matrix1(object):
    def __init__(self,matrix):
        matrix.replace(' ','')
        try:
            Try = Matrix1.Str2Mat(matrix.replace(' ',''))
            a = Matrix1.Str2Mat(matrix.replace(' ',''))
            for x in range(len(a)):
                if len(a) == 1:
                    continue
                elif len(a[1]) == len(a[1:][x-1]):
                    for x in a[1:]:
                        for y in x:
                            if type(y) == int or float or complex:
                                continue
                            else:
                                raise MatrixSyntaxError
                else:
                    raise MatrixSyntaxError
            self.matrix = a
        except Exception:
            raise MatrixSyntaxError

    def CanBeExcute(A,B):
        _a = A[1:]
        _b = B[1:]
        if len(_a) == len(_b) and len(_a[0]) == len(_b[0]):
            return True
        else:
            return False

    def CanBeExcute2(A,B):
        _a = A[1:]
        _b = B[1:]
        if len(_a[0]) == len(_b):
            return True
        else:
            return False
    def ww(s):
        if s[0] == '(' and s[-1] == ')':
            return s[1:-1]
        else:
            return s

    def W(A):
        A = Matrix1.ww(A)
        if 'j' in A:
            if 'j+' in A:
                if A[0] == '-':
                    x = A[0:A.index('+')]
                    y = A[A.index('+')+1:]
                    e = y  + x
                    return complex(e)
                else:
                    x = A[0:A.index('+')]
                    y = A[A.index('+')+1:]
                    e = y + '+' + x
                    return complex(e)
            elif 'j-' in A:
                if A[0] == '-':
                    x = A[0:A.index('j-')+1]
                    y = A[A.index('j-')+1:]
                    e = y + x
                    return complex(e)
                else:
                    x = A[0:A.index('-')]
                    y = A[A.index('-')+1:]
                    e = '-' + y + '+' + x
                    return complex(e)
            else:
                return complex(A)
        elif '.' in A:
            return float(A)
        else:
            return int(A)

    def V(A):
        A = str(A)
        if 'j)' in A:
            if '-0' in A:
                if A[3:-1] =='-0j':
                    return '0'
                elif A[3] == '+':
                    return A[4:-1]
                else:
                    return A[3:-1]
            elif '+0j' in A:
                return A[1:-1].replace('+0j','')
            else:
                return A[1:-1]
        elif A == '0j' or A == '-0j':
            return '0'
        elif A == '0.0':
            return '0'
        elif A == '-0.0':
            return '0'
        else:
            return A

    def IsStandard(A):
        if A[0] == 0:
            return True
        else:
            return False

    def Str2Mat(s):
        if s[0] == '[' and s[-1] == ']':
            if len(s) < 3:
                return [0]
            else:
                S = [0]
                rows = s[1:-1].split(';')
                for x in rows:
                    row1 = x.split(',')
                    row2 = []
                    for element in row1:
                        element = Matrix1.W(element)
                        row2.append(element)
                    S.append(row2)
                return Matrix1.change(S)
        else:
            S = [1]
            for x in range(int(s[0])):
                S.append([])
                for y in range(int(s[2])):
                    S[x+1].append(0)
            s1 = s[4:-2]
            s2 = s1.split('),')
            if len(s2[0]) < 5:
                return S
            else:
                for row in s2:
                    s3 = row.split(',')
                    S[int(s3[0][1])][int(s3[1])-1] = Matrix1.W(s3[2])
                LL = Matrix1.change(S)
                return LL


    def Mat2StrStandard(A):
        A = A[1:]
        if A == []:
            return '[]'
        else:
            l = ''
            for x in A:
                l = l + '['
                for y in x:
                    l  = l + Matrix1.V(y) + ','
                l = l + ']'
            l = l.replace(',][',';')
            return l.replace(',]',']')

    def Mat2StrSparse(A):
        A = A[1:]
        if A == []:
            return '0-0{}'
        else:
            mat1 = V(len(A)) + '-' + V(len(A[0])) + '{'
            for x in range(len(A)):
                for y in A[x]:
                    if y == 0:
                        continue
                    else:
                        mat1 = mat1 + '('+ V(x+1) + ',' + V(A[x].index(y) + 1) +',' + V(y)
                    mat1  = mat1 + ')' +','
            mat1 = mat1 + '}'
            return mat1.replace(',}','}')

    def Sparse2Standard(A):
        if not IsStandard(A):
            A[0] = 0
            return A
        else:
            return A

    def Standard2Sparse(A):
        if IsStandard(A):
            A[0] = 1
            return A
        else:
            return A

    def MatAdd(A, B):
        A = A[1:]
        B = B[1:]
        Mat1 = [0]
        for (row1, row2) in zip(A, B):
            Mat2 = []
            for (e1, e2) in zip(row1, row2):
                Mat2.append(e1 + e2)
            Mat1.append(Mat2)
        return Matrix1.change(Mat1)

    def MatSub(A,B):
        A = A[1:]
        B = B[1:]
        Mat1 = [0]
        for (row1,row2) in zip(A,B):
            Mat2 = []
            for (e1,e2) in zip(row1,row2):
                Mat2.append(e1-e2)
            Mat1.append(Mat2)
        return Matrix1.change(Mat1)

    def MatScalarMul(A, c):
        A = A[1:]
        Mat1 = [0]
        for row in A:
            Mat2 = []
            for element in row:
                Mat2.append(element*c)
            Mat1.append(Mat2)
        return Matrix1.change(Mat1)

    def MatScalardiv(A,c):
        A = A[1:]
        Mat1 = [0]
        for row in A:
            Mat2 = []
            for element in row:
                Mat2.append(element/c)
            Mat1.append(Mat2)
        return Matrix1.change(Mat1)

    def MatTransposition(A):
        s = A[0]
        A = A[1:]
        Mat1 = [s]
        for x in range(len(A[0])):
            Mat2 = []
            for y in range(len(A)):
                Mat2.append(A[y][x])
            Mat1.append(Mat2)
        return Mat1

    def transposition(self):
        return Matrix1.Mat2StrStandard(Matrix1.MatTransposition(self.matrix))

    def MatEq(A,B):
        if A[1:] == B[1:]:
            return True
        else:
            return False

    def MatMul(A,B):
        A = A[1:]
        B = B[1:]
        Result = [0]
        for row in A:
            mat1 = []
            for y in range(len(B[0])):
                element = 0
                for x in range(len(B)):
                    element = element + row[x] * B[x][y]
                    # global element
                mat1.append(element)
            Result.append(mat1)
        return Matrix1.change(Result)

    def isIdentity(self):
        import copy
        a = self.matrix[1:]
        b = copy.deepcopy(a)
        for row in b:
            if row[b.index(row)] == 1:
                row[b.index(row)] = 0
                for x in row:
                    if x == 0:
                        continue
                    else:
                        return False
            else:
                return False
        return True

    def isSquare(self):
        if len(self.matrix) == 1:
            return True
        elif len(self.matrix)-1 == len(self.matrix[1]):
            return True
        else:
            return False

    def other(M):
        if isinstance(M[0],list):
            return M
        else:
            return M[1:]

    def Yuzi(M,row,element):
        import copy
        a = Matrix1.other(M)
        b = copy.deepcopy(a)
        del b[row]
        for x in b:
            del x[element]
        return b





    def Determinant(M):
        a =Matrix1.other(M)
        if len(a) == len(a[0]):
            if len(a) == 1:
                return a[0][0]
            else:
                R = 0
                row = 0
                element = 0
                for x in a:
                    while element < len(a):
                        b = Matrix1.Yuzi(a,row,element)
                        R = R + (-1)**(row - element)*a[row][element]*Matrix1.Determinant(b)
                        element = element + 1
                    return R
        else:
            raise MatrixSyntaxError

    def determinant(self):
        a = self.matrix
        b = Matrix1.Determinant(a)
        return Matrix1.change2(b)


    def inverse(self):
        import copy
        if len(self.matrix) == 1:
            return Matrix1(Matrix1.Mat2StrStandard(Matrix1.Str2Mat(Matrix1.Mat2StrStandard([0]))))
        elif len(self.matrix) == len(self.matrix[1])+1:
            a = self.matrix[1:]
            if len(a) == 1 and a[0][0] != 0:
                x = 1/a[0][0]
                b = [0]
                b.append([x])
                return Matrix1(Matrix1.Mat2StrStandard(Matrix1.Str2Mat(Matrix1.Mat2StrStandard(b))))
            elif len(a) >= 2:
                if Matrix1.Determinant(a) != 0:
                    L = []
                    for row in range(len(a)):
                        LL = []
                        for element in range(len(a)):
                            K = (-1)**(row+element+2) * Matrix1.Determinant(Matrix1.Yuzi(a,row,element))
                            LL.append(K)
                        L.append(LL)
                    B = [0]
                    B.extend(L)
                    D = Matrix1.MatTransposition(B)
                    E = Matrix1(Matrix1.Mat2StrStandard(Matrix1.Str2Mat(Matrix1.Mat2StrStandard(D))))
                    Y = E/Matrix1.Determinant(a)
                    return Matrix1(Matrix1.Mat2StrStandard(Matrix1.Str2Mat(Matrix1.Mat2StrStandard(Y.matrix))))
                else:
                    raise MatrixSyntaxError
            else:
                raise MatrixSyntaxError
        else:
            raise MatrixSyntaxError

    def change(s):
        for x in s:
            if isinstance(x,int):
                continue
            elif isinstance(x,list):
                for y in x:
                    if isinstance(y,complex):
                        if y == 0:
                            s[s.index(x)][x.index(y)] = 0
                        elif str(complex(y)) != str(y):
                            if len(str(complex(y))) <= len(str(y)):
                                s[s.index(x)][x.index(y)] = complex(y)
                            else:
                                s[s.index(x)][x.index(y)] = y
                    elif isinstance(y,float):
                        if y == 0:
                            s[s.index(x)][x.index(y)] = 0
                        elif int(y) == y:
                            s[s.index(x)][x.index(y)] = int(y)
                        else:
                            s[s.index(x)][x.index(y)] = y
                    else:
                        s[s.index(x)][x.index(y)] = y
        return s

    def change2(s):
        if isinstance(s,complex):
            if s == 0:
                return 0
            elif str(complex(s)) != str(s):
                if len(str(complex(s))) <= len(str(s)):
                    return complex(s)
                else:
                    return s
            else:
                return complex(s)
        if isinstance(s,float):
            if int(s) == s:
                return int(s)
            else:
                return s
        else:
            return s


    def __str__(self):
        return Matrix1.Mat2StrStandard(self.matrix)

    def __add__(self,another):
        if isinstance(another,Matrix1):
            a = self.matrix
            b = another.matrix
            if Matrix1.CanBeExcute(a,b):
                c = Matrix1.MatAdd(a,b)
                return Matrix1(Matrix1.Mat2StrStandard(Matrix1.Str2Mat(Matrix1.Mat2StrStandard(c))))
            else:
                raise MatrixSyntaxError
        else:
            raise MatrixSyntaxError

    def __sub__(self,another):
        if isinstance(another,Matrix1):
            a = self.matrix
            b = another.matrix
            if Matrix1.CanBeExcute(a,b):
                c = Matrix1.MatSub(a,b)
                return Matrix1(Matrix1.Mat2StrStandard(Matrix1.Str2Mat(Matrix1.Mat2StrStandard(c))))
            else:
                raise MatrixSyntaxError
        else:
            raise MatrixSyntaxError

    def __mul__(self,another):
        a = self.matrix
        if isinstance(another,Matrix1):
            b = another.matrix
            if Matrix1.CanBeExcute2(a,b):
                c = Matrix1.MatMul(a,b)
                return Matrix1(Matrix1.Mat2StrStandard(Matrix1.Str2Mat(Matrix1.Mat2StrStandard(c))))
            else:
                raise MatrixSyntaxError
        else:
            c = Matrix1.MatScalarMul(a,another)
            return Matrix1(Matrix1.Mat2StrStandard(Matrix1.Str2Mat(Matrix1.Mat2StrStandard(c))))

    def __truediv__(self,another):
        a = self.matrix
        c = Matrix1.MatScalardiv(a,another)
        return Matrix1(Matrix1.Mat2StrStandard(Matrix1.Str2Mat(Matrix1.Mat2StrStandard(c))))

    def __pow__(self,another):
        import copy
        a = self.matrix
        if len(a) == len(a[1])+1:
            if isinstance(another,Matrix1):
                raise MatrixSyntaxError
            else:
                b = copy.deepcopy(a)
                for x in range(another-1):
                    a = Matrix1.MatMul(a,b)
                return Matrix1(Matrix1.Mat2StrStandard(Matrix1.Str2Mat(Matrix1.Mat2StrStandard(a))))
        else:
            raise MatrixSyntaxError



    def __getitem__(self,key):
        a = self.matrix
        if isinstance(key,int):
            A = [0]
            A.append(a[key+1])
            return Matrix1(Matrix1.Mat2StrStandard(Matrix1.Str2Mat(Matrix1.Mat2StrStandard(A))))
        elif isinstance(key,tuple):
            if isinstance(key[0],int):
                return a[key[0]+1][key[1]]
            elif isinstance(key[0],slice):
                b = a[1:]
                rows = b[key[0]]
                L = [0]
                for element in rows:
                    L.append(element[key[1]])
                return Matrix1(Matrix1.Mat2StrStandard(Matrix1.Str2Mat(Matrix1.Mat2StrStandard(L))))
            else:
                raise MatrixSyntaxError
        else:
            raise MatrixSyntaxError




    def __setitem__(self,key,value):
        if isinstance(key,tuple):
            if isinstance(key[0],int):
                self.matrix[key[0]+1][key[1]] = value
            elif isinstance(key[0],slice) and isinstance(value,Matrix1):
                b = self.matrix[1:]
                rows = b[key[0]]
                L = [0]
                for element in rows:
                    L.append(element[key[1]])
                if len(value.matrix) == len(L) and len(value.matrix[1]) == len(L[1]):
                    LL = []
                    for row in rows:
                        row[key[1]] = value.matrix[rows.index(row)+1]
                        LL.append(row)
                    self.matrix[1:][key[0]] = LL
                else:
                    raise MatrixSyntaxError
            else:
                raise MatrixSyntaxError
        elif isinstance(key,int) and isinstance(value,Matrix1) and len(value.matrix[1]) == len(self.matrix[key+1]):
            self.matrix[key+1] = value.matrix[1]
        else:
            raise MatrixSyntaxError


    def __neg__(self):
        A = Matrix1.other(self.matrix)
        L = [0]
        for row in A:
            LL = []
            for element in row:
                LL.append(-element)
            L.append(LL)
        return Matrix1(Matrix1.Mat2StrStandard(Matrix1.Str2Mat(Matrix1.Mat2StrStandard(L))))

    def __eq__(self,another):
        if isinstance(another,Matrix1):
            if another.matrix == self.matrix:
                return True
            else:
                return False
        else:
            return False

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

class Stack:
    def __init__(self):
        self.items = []
    def is_empty(self):
        return self.items == []
    def push(self, item):
        self.items.insert(0, item)
    def pop(self):
        return self.items.pop(0)
    def peek(self):
        return self.items[0]
    def size(self):
        return len(self.items)

class Matrix(object):
    def __init__(self,Matrix,code=None):
        try:
            if code == "prefix":
                self.matrix = self.DoPrefix(Matrix)
            elif code == None:
                if '*'not in Matrix and '/'not in Matrix and '-' not in Matrix and '+' not in Matrix:
                    self.matrix = Matrix1(Matrix)
                else:
                    self.matrix = self.DoPrefix(Matrix)
            elif code == "infix":
                self.matrix = self.DoInfix(Matrix)
            elif code == "postfix":
                self.matrix = self.DoPostfix(Matrix)
            else:
                raise MatrixSyntaxError
        except Exception:
            raise MatrixSyntaxError

    def ww(s):
        if s[0] == '(' and s[-1] == ')':
            return s[1:-1]
        else:
            return s

    def W(A):
        A = Matrix.ww(A)
        if 'j' in A:
            if 'j+' in A:
                if A[0] == '-':
                    x = A[0:A.index('+')]
                    y = A[A.index('+')+1:]
                    e = y  + x
                    return complex(e)
                else:
                    x = A[0:A.index('+')]
                    y = A[A.index('+')+1:]
                    e = y + '+' + x
                    return complex(e)
            elif 'j-' in A:
                if A[0] == '-':
                    x = A[0:A.index('j-')+1]
                    y = A[A.index('j-')+1:]
                    e = y + x
                    return complex(e)
                else:
                    x = A[0:A.index('-')]
                    y = A[A.index('-')+1:]
                    e = '-' + y + '+' + x
                    return complex(e)
            else:
                return complex(A)
        elif '.' in A:
            return float(A)
        else:
            return int(A)

    def MakeStandard(M1):#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        LJust1 = []
        L_Matrix = []
        for i in M1:
            LJust1.append(i)
        for j in LJust1:
            if j == '[':
                Mat1 = LJust1[LJust1.index('['):LJust1.index(']')+1]
                StrMat1 = ''.join(Mat1)
                RealMat = Matrix1(StrMat1)
                LJust1[LJust1.index('['):LJust1.index(']')+1] = ['$']*len(Mat1)
                L_Matrix.append(RealMat)
            elif j == '(':
                element = LJust1[LJust1.index('('):LJust1.index(')')+1]
                StrElement = ''.join(element)
                try:
                    RealElement = Matrix.W(StrElement)
                    LJust1[LJust1.index('('):LJust1.index(')')+1] = ['$']*len(element)
                    L_Matrix.append(RealElement)
                except Exception:
                    LJust1[LJust1.index(j)] = '$'
                    L_Matrix.append(j)
            elif j in '0123456789' or j == 'T':
                L_Matrix.append(j)
            elif j in '+-*/':
                L_Matrix.append(j)
            elif j == ')':
                L_Matrix.append(j)
                LJust1[LJust1.index(j)] = '$'
            elif j == '$':
                continue
            else:
                MatrixSyntaxError
        return L_Matrix


    def DoPostfix(self,postfix_expr):
        operand_stack = Stack()
        token_list = Matrix.MakeStandard(postfix_expr)
        for token in token_list:
            if isinstance(token,int) or isinstance(token,complex) or isinstance(token,float):
                operand_stack.push(token)
            elif isinstance(token,Matrix1):
                operand_stack.push(token)
            elif token == 'T':
                operandT = operand_stack.pop()
                result = Matrix.do_T(operandT)
                operand_stack.push(result)
            else:
                operand2 = operand_stack.pop()
                operand1 = operand_stack.pop()
                if isinstance(operand2,str):
                    operand2 = Matrix1(operand2)
                elif isinstance(operand1,str):
                    operand1 = Matrix1(operand1)
                result = Matrix.do_math(token, operand1, operand2)
                operand_stack.push(result)
        return operand_stack.pop()

    def DoPostfix2(postfix_expr):
        operand_stack = Stack()
        token_list = postfix_expr
        for token in token_list:
            if isinstance(token,int) or isinstance(token,complex) or isinstance(token,float):
                operand_stack.push(token)
            elif isinstance(token,Matrix1):
                operand_stack.push(token)
            elif token == 'T':
                operandT = operand_stack.pop()
                result = Matrix.do_T(operandT)
                operand_stack.push(result)
            else:
                operand2 = operand_stack.pop()
                operand1 = operand_stack.pop()
                if isinstance(operand2,str):
                    operand2 = Matrix1(operand2)
                elif isinstance(operand1,str):
                    operand1 = Matrix1(operand1)
                result = Matrix.do_math(token, operand1, operand2)
                operand_stack.push(result)
        return operand_stack.pop()



    def DoPostfix3(postfix_expr):
        try:
            operand_stack = Stack()
            token_list = postfix_expr
            for token in token_list:
                if isinstance(token,int) or isinstance(token,complex) or isinstance(token,float):
                    operand_stack.push(token)
                elif isinstance(token,Matrix1):
                    operand_stack.push(token)
                elif token == 'T':
                    operandT = operand_stack.pop()
                    result = Matrix.do_T(operandT)
                    operand_stack.push(result)
                else:
                    operand2 = operand_stack.pop()
                    operand1 = operand_stack.pop()
                    if isinstance(operand2,str):
                        operand2 = Matrix1(operand2)
                    elif isinstance(operand1,str):
                        operand1 = Matrix1(operand1)
                    result = Matrix.do_math(token, operand1, operand2)
                    operand_stack.push(result)
            return True
        except Exception:
            return False 


    def do_T(A):
        return A.transposition()

    def do_math(op, op1, op2):
        if op == "*":
            if isinstance(op2,Matrix1) and isinstance(op1,int):
                return op2 * op1
            elif isinstance(op2,Matrix1) and isinstance(op1,complex):
                return op2 * op1
            elif isinstance(op2,Matrix1) and isinstance(op1,float):
                return op2 * op1
            elif isinstance(op1,Matrix1) and isinstance(op2,Matrix1):
                return op1 * op2
            else:
                return op1 * op2
        elif op == "/":
            if isinstance(op2,Matrix1) and isinstance(op1,int):
                return op2 / op1
            elif isinstance(op2,Matrix1) and isinstance(op1,complex):
                return op2 / op1
            elif isinstance(op2,Matrix1) and isinstance(op1,float):
                return op2 / op1
            elif isinstance(op1,Matrix1) and isinstance(op2,Matrix1):
                return op1 / op2
            else:
                return op1 / op2
        elif op == "+":
            return op1 + op2
        elif op == '-':
            return op1 - op2
        else:
            raise MatrixSyntaxError

    def IsItReal(A):
        if isinstance(A,Matrix1) or isinstance(A,int) or isinstance(A,complex) or isinstance(A,float):
            return True

    def infix_to_postfix(infix_expr):
        prec = {}
        prec["T"] = 4
        prec["*"] = 3
        prec["/"] = 3
        prec["+"] = 2
        prec["-"] = 2
        prec["("] = 1
        op_stack = Stack()
        postfix_list = []
        token_list = infix_expr
        for token in token_list:
            if Matrix.IsItReal(token):
                postfix_list.append(token)
            elif token == '(':
                op_stack.push(token)
            elif token == ')':
                top_token = op_stack.pop()
                while top_token != '(':
                    postfix_list.append(top_token)
                    top_token = op_stack.pop()
            else:
                while (not op_stack.is_empty()) and (prec[op_stack.peek()] >= prec[token]):
                    postfix_list.append(op_stack.pop())
                op_stack.push(token)
        while not op_stack.is_empty():
            postfix_list.append(op_stack.pop())
        LLL = []
        for i in postfix_list:
            if i == '(':
                continue
            else:
                LLL.append(i)
        return LLL

    def DoInfix(self,infix_expr):
        infix_expr2 = Matrix.MakeStandard(infix_expr)
        if Matrix.DoPostfix3(infix_expr2):
            raise MatrixSyntaxError
        else:
            infix = Matrix.infix_to_postfix(infix_expr2)
            return Matrix.DoPostfix2(infix)

    def DoPrefix(self,prefix_expr):
        operand_stack = Stack()
        token_list = Matrix.MakeStandard(prefix_expr)
        for i in range(-1,-len(token_list)-1,-1):
            token = token_list[i]
            if isinstance(token,int) or isinstance(token,complex) or isinstance(token,float):
                operand_stack.push(token)
            elif isinstance(token,Matrix1):
                operand_stack.push(token)
            elif token == 'T':
                operandT = operand_stack.pop()
                result = Matrix.do_T(operandT)
                operand_stack.push(result)
            else:
                operand2 = operand_stack.pop()
                operand1 = operand_stack.pop()
                if isinstance(operand2,str):
                    operand2 = Matrix1(operand2)
                elif isinstance(operand1,str):
                    operand1 = Matrix1(operand1)
                result = Matrix.do_math(token, operand2, operand1)
                operand_stack.push(result)
        return operand_stack.pop()


    def __str__(self):
        return str(self.matrix)



# AAA = Matrix('[1,1;2,2]')
# A = Matrix('[1,2;2,2](41)[1,2;3,4]T*+','postfix')

# B = Matrix('[1,1;1,2] [1,1;1,0] + [0,1;2,0][0,0;1,1]+*','postfix')
# C = Matrix('(90)*(([1,1;1,2]+[1,1;1,0])*(((90)))*([0,1;2,0]+[0,0;1,1]))','infix')
# D = Matrix('*+ [1,1;1,2] [1,1;1,0]+[0,1;2,0][0,0;1,1]','prefix')
# E = Matrix('*  + [1,1;2,2] [0,0;1,1]+[1,1;2,3][4,4;3,3]','prefix')
# print(AAA)
# print(A)
# print(B)
# print(C)
# print(D)
# print(E)
#AA = Matrix('[1,2;2,2](41)[1,2;3,4]T*+','infix')
#print(AA)
