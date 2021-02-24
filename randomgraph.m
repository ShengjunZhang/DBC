% NAME randomgraph - generates 2-D random graphs
% SYNOPSIS
% [error,A,xy]=randomgraph(n,radius)
% DESCRIPTION
% Generates strongly connected random graphs

 
% INPUTS
% n --->  number of nodes
% r --->  connectivity radius

%%Created 10.1.2006
%%by Scaglione-Yildiz;Lausanne, Switzerland 
%%Last modified 4.1.2007
%%Ithaca, NY

function [error,A,degree, xy, L]=randomgraph(n,radius)
loop=1;
count=0;
error=0;
s = rng('default');
% rand('state',seed);
while(loop && count<10)
    xy=rand(n,2);
    %--> distance matrix Md(i,j)=sqrt( (x(i)-x(j))^2+(y(i)-y(j))^2);
          % of all pairs of nodes
    Md=sqrt((xy(:,1)*ones(1,n)-ones(n,1)*xy(:,1).').^2+(xy(:,2)*ones(1,n)-ones(n,1)*xy(:,2).').^2);
    %---> Adjacency matrix
    A=((Md+2*radius*eye(n))<radius)*eye(n);
    
    degree=A*ones(n,1);
    %---> Laplacian
    L=diag(degree)-A;
    %---> connectivity check
    if(rank(L)==n-1)
        loop=0;
    end
%     close;
%     figure, gplot(A,xy), hold on, plot(xy(:,1),xy(:,2),'ro'), hold off,
%     xlabel('x'),ylabel('y')
%     rank(L)
%     pause;
    count=count+1;
end

%---> If number of trials are greater than 10, exit
if(count==10)
    error=1;
end


