function [error,A,degree, xy, L]=linegraph(n,radius)
loop=1;
count=0;
error=0;
while(loop && count<10)
    xy = [(1:n)', ones(n,1)];
    %--> distance matrix of all pairs of nodes
    Md=sqrt( (xy(:,1)*ones(1,n)-ones(n,1)*xy(:,1).').^2+(xy(:,2)*ones(1,n)-ones(n,1)*xy(:,2).').^2);
    %---> Adjacency matrix
    A=((Md+2*radius*eye(n))<radius)*eye(n);
    
    degree=A*ones(n,1);
    %---> Laplacian
    L=diag(degree)-A;
    %---> connectivity check
    if(rank(L)==n-1)
        loop=0;
    end
    count=count+1;
end
%---> If number of trials are greater than 10, exit
if(count==10)
    error=1;
end

end
