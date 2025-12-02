import math
import argparse
import random
from tqdm import tqdm

def make_expressions(random_state):

    import sympy as sp
    x, y, z = sp.symbols('x y z')
    primitives = [x,y,z]

    style = random_state.choice(['poly', 'frac', 'power', 'trig', 'mix'])

    def rand_poly(depth=2):
        deg = random_state.randint(1,3)
        terms =[]
        for _ in range(random_state.randint(1,4)):
            coeff = random_state.randint(-9,9)
            var = random_state.choice(primitives)
            power = random_state.randint(0, deg)
            if power == 0:
                terms.append(sp.Integer(coeff))
            else:
                terms.append(coeff * var**power)
        expr = sum(terms)
        return expr
        
    if style == 'poly':
        expr = rand_poly()
    elif style == 'frac':
        num = rand_poly()
        den = rand_poly()

        if den == 0:
            den == 1

        if isinstance(num, den) and isinstance(den, sp.Integer):
            expr = sp.Rational(num, den)
        else:
            expr = num/den

    elif style == 'power':
        base = rand_poly()
        exp = random_state.randint(2, 4)
        expr = base ** exp
    elif style == 'trig':
        func = random_state.choice([sp.sin, sp.cos, sp.tan])
        inner = rand_poly()
        expr = func(inner)
    elif style == 'mix':
        expr = rand_poly()
        if random_state.random() < 0.5:
            expr = expr + sp.Rational(random_state.randint(-5,5), random_state.randint(1,6))
        if random_state.random() < 0.3:
            expr = expr + sp.sin(rand_poly())

    if random_state.random() < 0.25:
        expr = expr + rand_poly()
    
    return expr

def to_ascii(expr):
    s = str(expr)
    s = s.replace('**', '^')
    s = s.replace('Rational(', '')
    s = s.replace(')', '')
    return s

def generate(n, out_path, seed=42):
    random_state = random.Random(seed)
    import sympy as sp
    

    with open(out_path, 'w', encoding='utf-8') as f:
        for _ in tqdm(range(n), desc = "Generating synthetic math-> LaTeX"):
            expr = make_expressions(random_state)
            try:
                latex_str = sp.latex(expr)
            except Exception:
                latex_str = str(expr)
            
            ascii_in = to_ascii(expr)

            f.write("Instruction:\n")
            f.write("Convert the following mathematical expression into LaTeX.\n")
            f.write("Input:\n")
            f.write(ascii_in.strip()+"\n")
            f.write("Solution:\n")
            f.write(f"{latex_str}$\n")
            f.write("Final Answer:\n")
            f.write(latex_str.strip() + "\n\n")
    print(f"Saved {n} synthetic examples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic math -> LaTeX examples")
    parser.add_argument("--n", type=int, default=2500, help="Number of synthetic examples to generate (default 10000)")
    parser.add_argument("--out", type=str, default="training_data/synthetic_math_latex.txt", help="Output file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    generate(args.n, args.out, seed=args.seed)
