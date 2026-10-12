#include <iostream>
int main() {
    int n; std::cin >> n;
    for (int i=0; i<n; ++i) {
        int x,y,r; std::cin >> x >> y >> r;
        std::cout << x << " " << y << " " << x+1 << " " << y+1 << "\n";
    }
}
