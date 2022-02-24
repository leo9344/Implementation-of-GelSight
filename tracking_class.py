# from re import M
import numpy as np
import time

MAXN = 30
MAXM = 30
MAXNM = 900
class tracking_class():
    def __init__(self, N_, M_, fps_, x0_, y0_, dx_, dy_) -> None:
        self.N = N_
        self.M = M_
        self.NM = N_ * M_
        self.x_0 = x0_
        self.y_0 = y0_
        self.dx = dx_
        self.dy = dy_

        self.Row = np.zeros((MAXNM),dtype=int)
        self.Col = np.zeros((MAXNM),dtype=int)
        self.Dist = np.zeros((MAXNM, MAXNM),dtype=int)
        self.done = np.zeros((MAXN),dtype=int)
        self.occupied = np.zeros((MAXN, MAXM), dtype=int)
        self.first = np.zeros((MAXN),dtype=int)
        self.fps = fps_
        self.degree = np.zeros((MAXNM, MAXNM),dtype=float)

        self.dmin, self.dmax, self.theta = (0.5*self.dx)**2,(1.8*self.dx)**2,70
        self.moving_max = self.dx * 2
        self.cost_threshold = 15000 * (self.dx / 21)* (self.dx / 21)
        self.flow_difference_threshold = self.dx * 0.8

        self.time_st = 0
        self.n = 0 # num of detected markers

        self.flag_record = 1
        self.K1, self.K2 = 0.1, 1.0
        self.minf = -1
        self.MinRow, self.MinCol, self.MinOccupied = np.zeros((MAXNM),dtype=int),\
                                                     np.zeros((MAXNM),dtype=int),\
                                                     np.zeros((MAXN, MAXM), dtype=int)
        self.O, self.D, self.C, self.MinD = np.zeros((MAXN, MAXM, 2), dtype=float),\
                                            np.zeros((MAXN, MAXM,2), dtype=float),\
                                            np.zeros((MAXNM,2),dtype=float),\
                                            np.zeros((MAXN, MAXM,2), dtype=float)
 
        for i in range(self.N):
            for j in range(self.M):
                self.O[i,j,0] = x0_ + j * dx_
                self.O[i,j,1] = y0_ + i * dy_
        print("init O")
        # x, y, id
    def sum(self,i):
        res = (i[0]*i[0] + i[1]*i[1])
        return res
    def precessor(self, i, j):
        # int index i,j
        res = (self.degree[i,j] <= self.theta
            and self.degree[i,j] >= -self.theta
            and self.Dist[i,j] <= self.dmax
            and self.Dist[i,j] >= self.dmin)
        return res

    def init(self, centers):
        i, j = 0,0
        shape = np.shape(centers)
        self.n = shape[0]
    
        for i in range(0, self.n):
            self.C[i,0] = centers[i][0]
            self.C[i,1] = centers[i][1]
            # self.C[i,2] = i
        # print(self.C)
        self.done[:] = 0
        self.occupied[:,:] = -1

        self.minf = -1

        ## sort
        order = np.lexsort( (self.C[:self.n,1], self.C[:self.n,0]) )
        order_list = np.zeros((self.n,2),dtype=int)
        for i in range(self.n):
            order_list[order[i]] = self.C[order[i]]
        self.C[:self.n] = order_list
        # print(self.C[:self.n,0])
        ## correct
        for i in range(0, self.n):
            for j in range(0, i):
                self.Dist[i,j] = (self.C[i,0]-self.C[j,0])**2 + (self.C[i,1]-self.C[j,1])**2

                temp = np.abs(self.C[i,1] - self.C[j,1])/np.sqrt(self.Dist[i,j])
                if temp > 1:
                    temp = 1
                if temp < -1:
                    temp = -1
                self.degree[i,j] = np.arcsin(temp) *180.0 / np.pi

    def run(self):
        missing, spare = 0, 0

        self.time_st = time.time()

        missing = self.NM - self.n
        spare = self.n - self.NM
        missing = 0 if missing < 0 else missing
        spare = 0 if spare < 0 else spare

        self.dfs(0,0,missing,spare) 
        for i in range(1,4):
            if self.minf == -1:
                self.done[:] = 0
                self.occupied[:,:] = -1
                self.dfs(0, 0, missing+1 ,spare+1)

        if self.flag_record == 1:
            self.flag_record = 0
            print("O init")
            # print(self.O)
            for i in range(0,self.n):
                self.O[self.MinRow[i], self.MinCol[i], 0] = self.C[i,0]
                self.O[self.MinRow[i], self.MinCol[i], 1] = self.C[i,1]
            print(self.O[:,:,0]) # error. Probably MinRow and MinCol have not be calculated

    def get_flow(self):
        Ox = np.zeros((self.N, self.M))
        Oy = np.zeros((self.N, self.M))
        Cx = np.zeros((self.N, self.M))
        Cy = np.zeros((self.N, self.M))
        Occupied = np.zeros((self.N, self.M))
        for i in range(0,self.N):
            for j in range(0, self.M):
                Ox[i,j] = self.O[i,j,0]
                Oy[i,j] = self.O[i,j,1]
                Cx[i,j] = self.MinD[i,j,0]
                Cx[i,j] = self.MinD[i,j,1]
                Occupied[i,j] = self.MinOccupied[i,j]


        return (Ox, Oy, Cx, Cy, Occupied)
    
    def calc_cost(self,i):
        # i = int(i)
        c, cost, left, up, down = 0,0,0,0,0
        flow1 = [0,0]
        flow2 = [0,0]
        cost += self.K1 * self.sum(self.C[i] - self.O[self.Row[i], self.Col[i]])

        flow1 = self.C[i]-self.O[self.Row[i], self.Col[i]]
        if self.Col[i] > 0:
            left = self.occupied[self.Row[i], self.Col[i]-1]
            if left > -1:
                flow2 = self.C[left] - self.O[self.Row[i], self.Col[i]-1]
                c = self.sum(flow2 - flow1)
                if np.sqrt(c) >= self.flow_difference_threshold:
                    c = 1e8
                cost += self.K2 * c
        
        if self.Row[i] > 0 :
            up = self.occupied[self.Row[i]-1, self.Col[i]]
            if up > -1:
                flow2 = self.C[up] - self.O[self.Row[i]-1, self.Col[i]]
                c = self.sum(flow2 - flow1)
                # print(f"row[i] {c}")
                if np.sqrt(c) >= self.flow_difference_threshold:
                    c = 1e8
                cost += self.K2 * c
        
        if self.Row[i] < self.N - 1 :
            down = self.occupied[self.Row[i] + 1, self.Col[i]]
            if down > -1:
                flow2 = self.C[down] - self.O[self.Row[i]+1, self.Col[i]]
                c = self.sum(flow2-flow1)
                # print(f"row i N-1 {c}")
                # if c <0:
                    # print(f"{flow2-flow1}, {self.sum(flow)}")
                if np.sqrt(c) >= self.flow_difference_threshold:
                    c = 1e8
                cost += self.K2 * c

        return cost

    def infer(self):
        cost = 0
        boarder_nb = 0
        i, j, k, x, y, d, cnt, nx, ny, nnx, nny = 0,0,0,0,0,1,0,0,0,0,0

        dir = [[0,-1], [-1,0], [0,1], [1,0]]
        flow1 = np.array([0,0],dtype=float)
        flow2 = np.array([0,0],dtype=float)
        moving = np.array([0,0],dtype=float)

        for i in range(0,self.N):
            for j in range(0, self.M):
                if self.occupied[i,j] <= -1:
                    moving[0] = 0
                    moving[1] = 0
                    cnt = 0
                    for k in range(0,4):
                        nx = i + dir[k][0]
                        ny = j + dir[k][1]
                        nnx = nx + dir[k][0]
                        nny = ny + dir[k][1]
                        if nnx<0 or nnx >= self.N or nny<0 or nny>=self.M:
                            continue
                        if self.occupied[nx,ny] <= -1 or self.occupied[nnx][nny] <= -1 :
                            continue
                        moving += (self.C[self.occupied[nx, ny]] - self.O[nx,ny] + (self.C[self.occupied[nx, ny]] 
                                    - self.O[nx, ny] - self.C[self.occupied[nnx, nny]] + self.O[nnx, nny]));
                        cnt += 1
                    if cnt == 0:
                        for x in range(i-d,i+d+1):
                            for y in range(j-d, j+d+1):
                                if x<0 or x>=self.N or y<0 or y>=self.M :
                                    continue
                                if self.occupied[x,y] <= -1:
                                    continue
                                moving += self.C[self.occupied[x,y]] - self.O[x,y]
                                cnt += 1
                    if cnt == 0:
                        for x in range(i-d-1,i+d+2):
                            for y in range(j-d-1, j+d+2):
                                if x<0 or x>=self.N or y<0 or y>=self.M :
                                    continue
                                if self.occupied[x,y] <= -1:
                                    continue
                                moving += self.C[self.occupied[x,y]] - self.O[x,y]
                                cnt += 1
                    self.D[i,j] = self.O[i,j] + moving/(cnt+1e-6)
                    if j==0 and self.D[i,j,1] >= (self.O[i,j,1] - self.dy/2.0):
                        boarder_nb += 1
                    if j==self.N-1 and self.D[i,j,1] <= (self.O[i,j,1] + self.dy/2.0):
                        boarder_nb += 1
                    cost += self.K1 * self.sum(self.D[i,j] - self.O[i,j])

        if boarder_nb >= self.N-1:
            cost += 1e7
        
        for i in range(0, self.N):
            for j in range(0, self.M):
                if self.occupied[i,j] <= -1:
                    flow1 = self.D[i,j] - self.O[i,j]
                    for k in range(0,4):
                        nx = i + dir[k][0]
                        ny = j + dir[k][1]
                        if nx<0 or nx>self.N-1 or ny<0 or ny>self.M-1:
                            continue
                        if self.occupied[nx,ny] > -1:
                            flow2 = self.C[self.occupied[nx,ny]] - self.O[nx,ny]
                            cost += self.K2 * self.sum(flow2-flow1)
                        elif k<2 and self.occupied[nx,ny] <= -1:
                            flow2 = self.D[nx,ny] - self.O[nx,ny]
                            cost += self.K2 * self.sum(flow2-flow1)

        return cost

    def dfs(self,i,cost,missing,spare):

        if time.time()-self.time_st >= 1/self.fps:
            print("abnormal")
            return 
        # if (((float)(clock()-time_st))/CLOCKS_PER_SEC >= 1.0 / fps) return; For what?
        if cost >= self.minf and self.minf != -1:
            return 
        if cost >= self.cost_threshold:
            return
        j, k, count, flag, m, same_col = 0,0,0,0,0,0
        c = 0.0

        if i >= self.n:
            cost += self.infer()
            if cost<self.minf or self.minf == -1:
                self.minf = cost
                for j in range(0,self.n):
                    self.MinRow[j] = self.Row[j]
                    self.MinCol[j] = self.Col[j]
                    if self.Row[j]<0:
                        continue
                    self.D[self.Row[j], self.Col[j], 0] = self.C[j,0]
                    self.D[self.Row[j], self.Col[j], 1] = self.C[j,1]

                for j in range(0, self.N):
                    for k in range(0, self.M):
                        self.MinOccupied[j,k] = self.occupied[j,k]
                        self.MinD[j,k,0] = self.D[j,k,0]
                        self.MinD[j,k,1] = self.D[j,k,1]
            print("normal")
            return

        for j in range(i):
            if self.precessor(i, j):
                self.Row[i] = self.Row[j]
                self.Col[i] = self.Col[j] + 1
                count += 1
                if self.Col[i] >= self.M:
                    continue
                if self.occupied[self.Row[i], self.Col[i]] > -1:
                    continue
            if (self.Row[i] > 0 and self.occupied[self.Row[i]-1, self.Col[i]] > -1 
                and self.C[i,1] <= self.C[self.occupied[self.Row[i]-1, self.Col[i]],1]):
                continue
            if (self.Row[i] < self.N - 1 and self.occupied[self.Row[i]+1, self.Col[i]] > -1 
                and self.C[i,1] >= self.C[self.occupied[self.Row[i]+1, self.Col[i]],1]):
                continue
                
            vflag = 0
            for k in range(self.N):
                same_col = self.occupied[k,self.Col[i]]
                if (same_col > -1 and ((k < self.Row[i] and self.C[same_col,1] > self.C[i,1]) 
                    or (k > self.Row[i] and self.C[same_col,1] < self.C[i,1]))):
                    vflag = 1
                    break
            
            if vflag == 1:
                continue
            self.occupied[self.Row[i], self.Col[i]] = i

            c = self.calc_cost(i)
            self.dfs(i+1,cost+c,missing,spare)
            self.occupied[self.Row[i], self.Col[i]] = -1

        for j in range(self.N):
            if self.done[j] == 0:
                flag = 0
                for k in range(self.N):
                    if (self.done[k] and ((k < j and self.first[k] > self.C[i,1]) or (k > j and self.first[k] < self.C[i,1])) ):
                        flag = 1
                        break
                
                if flag == 1:
                    continue
                self.done[j] = 1
                self.first[j] = self.C[i,1]
                self.Row[i] = j
                self.Col[i] = 0

                self.occupied[self.Row[i], self.Col[i]] = i
                c = self.calc_cost(i)

                self.dfs(i+1, cost+c, missing, spare)
                self.done[j] = 0
                self.occupied[self.Row[i], self.Col[j]] = -1
        for m in range(1,missing+1):
            for j in range(self.N):
                if np.abs(self.C[i,1] - self.O[j,0,1] > self.moving_max):
                    continue
                for k in range(self.M-1, -1,-1):
                    if self.occupied[j,k] > -1:
                        break
                if k+m+1 >= self.M:
                    continue
                if np.sqrt(self.sum(self.C[i]-self.O[j,k+m+1])) > self.moving_max:
                    continue
                for t in range(1,m+1):
                    self.occupied[j,k+t] = -2
                
                self.Row[i] = j
                self.Col[i] = k + m + 1
                c = self.calc_cost(i)
                self.occupied[self.Row[i], self.Col[i]] = i

                self.dfs(i+1, cost+c, missing-m, spare) #forget to -m

                for t in range(1,m+2):
                    self.occupied[j,k+t] = -1
        
        if spare > 0 :
            self.Row[i] = -1
            self.Col[i] = -1
            self.dfs(i+1, cost, missing, spare-1)



                