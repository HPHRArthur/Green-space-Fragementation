#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urban green-space Monte Carlo + PuLP MILP prototype.

Inputs:
  green_map.png       green pixels = green space, other pixels = non-green
  planning_mask.png   white/light = change allowed; black/dark = frozen
  districts.png       grayscale integer district IDs (0 may be background; configurable)
  district_targets.csv district_id,min_green_fraction,max_loss_fraction,min_growth_fraction
Optional target columns may be blank. Fractions are relative to the district's total pixel
count, except max_loss_fraction which is a fraction of the district's initial green area.
All images must have identical dimensions and be aligned pixel-for-pixel.

This is a raster prototype, not a geospatially referenced GIS workflow.
"""
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import pulp
except ImportError:
    print("Missing PuLP. Install dependencies with: pip install pulp numpy pillow", file=sys.stderr)
    raise

def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)

def green_mask(rgb, green_threshold=35):
    # Green-dominant classifier. Tune for your image palette.
    r, g, b = rgb[:,:,0].astype(int), rgb[:,:,1].astype(int), rgb[:,:,2].astype(int)
    return (g >= 80) & (g-r >= green_threshold) & (g-b >= green_threshold)

def save_mask(mask, path):
    arr = np.where(mask, 0, 255).astype(np.uint8)
    # green pixels displayed green, non-green white
    rgb = np.full((*mask.shape,3), 255, dtype=np.uint8)
    rgb[mask] = np.array([0, 160, 0], dtype=np.uint8)
    Image.fromarray(rgb).save(path)

def read_targets(path):
    if not path:
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"district_id"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("district_targets.csv must contain a district_id column")
        for row in reader:
            did = int(row["district_id"])
            vals = {}
            for key in ("min_green_fraction", "max_loss_fraction", "min_growth_fraction"):
                raw = (row.get(key) or "").strip()
                vals[key] = None if raw == "" else float(raw)
                if vals[key] is not None and not (0 <= vals[key] <= 1):
                    raise ValueError(f"{key} must be between 0 and 1; district {did}")
            out[did] = vals
    return out

def neighbors4(mask):
    # Boolean cells adjacent (4-neighbor) to any green cell.
    edge = np.zeros_like(mask, dtype=bool)
    edge[1:,:] |= mask[:-1,:]
    edge[:-1,:] |= mask[1:,:]
    edge[:,1:] |= mask[:,:-1]
    edge[:,:-1] |= mask[:,1:]
    return edge & (~mask)

def component_count(mask, connectivity=4):
    # Flood-fill component count; used only for reporting, not as a MILP constraint.
    h,w=mask.shape
    seen=np.zeros_like(mask,dtype=bool)
    if connectivity == 8:
        offsets=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    else:
        offsets=[(-1,0),(1,0),(0,-1),(0,1)]
    count=0
    for y,x in zip(*np.where(mask & ~seen)):
        if seen[y,x]: continue
        count+=1; stack=[(int(y),int(x))]; seen[y,x]=True
        while stack:
            cy,cx=stack.pop()
            for dy,dx in offsets:
                ny,nx=cy+dy,cx+dx
                if 0<=ny<h and 0<=nx<w and mask[ny,nx] and not seen[ny,nx]:
                    seen[ny,nx]=True; stack.append((ny,nx))
    return count

def make_candidates(current, allowed, districts, growth_rate, loss_rate, rng, max_candidates, stochastic_split_rate):
    # Candidate additions must be edge-adjacent to existing green and allowed by the planning mask.
    grow = np.flatnonzero((neighbors4(current) & allowed).ravel())
    # Candidate removals are existing green pixels in allowed areas.
    lose = np.flatnonzero((current & allowed).ravel())
    # Randomly sample candidates to keep MILP manageable.
    grow_budget = min(len(grow), max_candidates, max(1, int(np.ceil(growth_rate * max(1, current.sum()) * 3))))
    loss_budget = min(len(lose), max_candidates, max(1, int(np.ceil(loss_rate * max(1, current.sum()) * 3))))
    if len(grow) > grow_budget: grow = rng.choice(grow, grow_budget, replace=False)
    if len(lose) > loss_budget: lose = rng.choice(lose, loss_budget, replace=False)
    # Candidate preference scores: random Monte Carlo utility, plus split-risk penalty.
    # Articulation / fragmentation is not imposed as a hard constraint in this prototype.
    grow_pref = rng.random(len(grow))
    loss_pref = rng.random(len(lose))
    if len(lose):
        # A random subset of loss candidates gets a fragmentation-risk penalty.
        risky = rng.random(len(lose)) < stochastic_split_rate
        loss_pref = loss_pref - risky.astype(float) * 2.0
    return grow.astype(np.int64), lose.astype(np.int64), grow_pref, loss_pref

def solve_year(current, allowed, districts, targets, growth_rate, loss_rate, rng,
               max_candidates=3000, stochastic_split_rate=0.1, solver_msg=False):
    h,w=current.shape
    n_green=int(current.sum())
    grow, lose, gpref, lpref = make_candidates(
        current, allowed, districts, growth_rate, loss_rate, rng,
        max_candidates, stochastic_split_rate
    )
    # Nominal target counts. They are targets, not guaranteed if candidates/constraints make them infeasible.
    desired_add=max(0, int(round(growth_rate*n_green)))
    desired_loss=max(0, int(round(loss_rate*n_green)))
    prob=pulp.LpProblem("UrbanGreenYear", pulp.LpMinimize)
    add={int(p):pulp.LpVariable(f"add_{int(p)}",cat="Binary") for p in grow}
    rem={int(p):pulp.LpVariable(f"rem_{int(p)}",cat="Binary") for p in lose}
    add_slack=pulp.LpVariable("add_shortfall", lowBound=0)
    loss_slack=pulp.LpVariable("loss_deviation", lowBound=0)
    # Objective: meet target growth; avoid excess losses; favor sampled high-utility growth locations.
    prob += (
        1000*add_slack
        + 20*loss_slack
        + pulp.lpSum((1.0-float(gpref[j]))*add[int(p)] for j,p in enumerate(grow))
        + pulp.lpSum((1.0-float(lpref[j]))*rem[int(p)] for j,p in enumerate(lose))
    )
    prob += pulp.lpSum(add.values()) + add_slack >= desired_add, "city_growth_target"
    # Loss target is treated as a soft target to avoid forcing losses; penalize deviation from nominal.
    # abs(total loss - desired_loss) <= loss_deviation
    total_loss=pulp.lpSum(rem.values())
    prob += total_loss - desired_loss <= loss_slack, "loss_deviation_upper"
    prob += desired_loss - total_loss <= loss_slack, "loss_deviation_lower"

    # District constraints. Values are fractions, based on each district's pixel count.
    for did in np.unique(districts):
        did=int(did)
        if did == 0:  # ID 0 is reserved for background by default
            continue
        dmask=(districts==did)
        dsize=int(dmask.sum())
        if dsize == 0: continue
        dgreen=int((current & dmask).sum())
        row=targets.get(did,{})
        d_add=[int(p) for p in grow if districts.ravel()[int(p)]==did]
        d_rem=[int(p) for p in lose if districts.ravel()[int(p)]==did]
        if row.get("min_green_fraction") is not None:
            minimum=int(np.ceil(row["min_green_fraction"]*dsize))
            prob += dgreen + pulp.lpSum(add[p] for p in d_add) - pulp.lpSum(rem[p] for p in d_rem) >= minimum, f"district_{did}_min_green"
        if row.get("min_growth_fraction") is not None:
            minimum_growth=int(np.ceil(row["min_growth_fraction"]*dsize))
            prob += pulp.lpSum(add[p] for p in d_add) >= minimum_growth, f"district_{did}_min_growth"
        if row.get("max_loss_fraction") is not None:
            max_loss=int(np.floor(row["max_loss_fraction"]*dgreen))
            prob += pulp.lpSum(rem[p] for p in d_rem) <= max_loss, f"district_{did}_max_loss"

    solver=pulp.PULP_CBC_CMD(msg=solver_msg)
    status=prob.solve(solver)
    status_name=pulp.LpStatus.get(status, str(status))
    if status_name not in ("Optimal", "Feasible"):
        # Return unchanged map and record failure rather than silently violate hard constraints.
        return current.copy(), {
            "status":status_name,"green_before":n_green,"green_after":n_green,
            "added":0,"lost":0,"components_before":None,"components_after":None,
            "candidate_growth":len(grow),"candidate_loss":len(lose),
            "objective":None,"note":"MILP infeasible/unsolved; map unchanged"
        }
    new=current.copy()
    flat=new.ravel()
    added=[p for p in grow if pulp.value(add[int(p)]) is not None and pulp.value(add[int(p)])>0.5]
    removed=[p for p in lose if pulp.value(rem[int(p)]) is not None and pulp.value(rem[int(p)])>0.5]
    flat[added]=True
    flat[removed]=False
    return new, {
        "status":status_name,"green_before":n_green,"green_after":int(new.sum()),
        "added":len(added),"lost":len(removed),"components_before":component_count(current),
        "components_after":component_count(new),"candidate_growth":len(grow),
        "candidate_loss":len(lose),"objective":float(pulp.value(prob.objective)),
        "note":""
    }

def main():
    ap=argparse.ArgumentParser(description="Monte Carlo + PuLP MILP urban green-space simulation")
    ap.add_argument("--input",required=True,help="Initial green map image")
    ap.add_argument("--planning-mask",required=True,help="White/light=allowed, black/dark=frozen")
    ap.add_argument("--districts",required=True,help="Grayscale district ID image; 0 is background")
    ap.add_argument("--district-targets",required=True,help="CSV with district target constraints")
    ap.add_argument("--output",default="results_milp")
    ap.add_argument("--growth-rate",type=float,default=0.03)
    ap.add_argument("--loss-rate",type=float,default=0.02)
    ap.add_argument("--years",type=int,default=10)
    ap.add_argument("--runs",type=int,default=20)
    ap.add_argument("--max-candidates",type=int,default=3000,help="Maximum candidate add/remove pixels per year")
    ap.add_argument("--split-risk-rate",type=float,default=0.10,help="Soft penalty probability for loss candidates; not a hard split model")
    ap.add_argument("--green-threshold",type=int,default=35)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--solver-msg",action="store_true")
    args=ap.parse_args()
    if not (0<=args.growth_rate<=1 and 0<=args.loss_rate<=1):
        ap.error("growth-rate and loss-rate must be between 0 and 1")
    if args.years<1 or args.runs<1 or args.max_candidates<1:
        ap.error("years, runs and max-candidates must be positive")
    rgb=load_rgb(args.input)
    initial=green_mask(rgb,args.green_threshold)
    planning=load_gray(args.planning_mask)
    allowed=planning>=128
    districts=load_gray(args.districts).astype(np.int32)
    if initial.shape!=allowed.shape or initial.shape!=districts.shape:
        ap.error("Input, planning mask and district image must have identical dimensions")
    targets=read_targets(args.district_targets)
    missing=set(targets)-set(np.unique(districts).tolist())
    if missing: print(f"Warning: target district IDs not present in raster: {sorted(missing)}")
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    (out/"maps").mkdir(exist_ok=True)
    save_mask(initial,out/"classified_initial.png")
    with open(out/"run_config.json","w",encoding="utf-8") as f:
        json.dump(vars(args),f,ensure_ascii=False,indent=2)
    all_rows=[]
    for run in range(args.runs):
        rng=np.random.default_rng(args.seed+run)
        current=initial.copy()
        save_mask(current,out/"maps"/f"run_{run+1:03d}_year_00.png")
        for year in range(1,args.years+1):
            current,info=solve_year(current,allowed,districts,targets,args.growth_rate,args.loss_rate,rng,
                                    args.max_candidates,args.split_risk_rate,args.solver_msg)
            info.update({"run":run+1,"year":year,"green_fraction":float(current.mean())})
            all_rows.append(info)
            save_mask(current,out/"maps"/f"run_{run+1:03d}_year_{year:02d}.png")
            print(f"run {run+1}/{args.runs} year {year}/{args.years}: {info['status']}, "
                  f"green={info['green_after']}, +{info['added']} -{info['lost']}")
    # CSV output
    fields=list(dict.fromkeys(k for row in all_rows for k in row.keys()))
    with open(out/"yearly_metrics_all_runs.csv","w",newline="",encoding="utf-8-sig") as f:
        wr=csv.DictWriter(f,fieldnames=fields); wr.writeheader(); wr.writerows(all_rows)
    # Summary by year
    summary=[]
    for year in range(1,args.years+1):
        rows=[r for r in all_rows if r["year"]==year]
        for metric in ("green_after","added","lost","components_after","green_fraction"):
            vals=np.array([r[metric] for r in rows if r.get(metric) is not None],dtype=float)
            if len(vals):
                summary.append({"year":year,"metric":metric,"mean":float(vals.mean()),
                                "p05":float(np.percentile(vals,5)),"median":float(np.median(vals)),
                                "p95":float(np.percentile(vals,95))})
    with open(out/"metrics_summary_by_year.csv","w",newline="",encoding="utf-8-sig") as f:
        wr=csv.DictWriter(f,fieldnames=["year","metric","mean","p05","median","p95"])
        wr.writeheader(); wr.writerows(summary)
    # Per-pixel year-end green probability across runs
    accum=np.zeros(initial.shape,dtype=np.float64)
    for run in range(args.runs):
        p=out/"maps"/f"run_{run+1:03d}_year_{args.years:02d}.png"
        arr=load_rgb(p)
        accum += green_mask(arr,args.green_threshold)
    prob_img=np.clip(accum/args.runs*255,0,255).astype(np.uint8)
    Image.fromarray(prob_img).save(out/"year_end_green_probability.png")
    consensus=(accum/args.runs)>=0.5
    save_mask(consensus,out/"year_end_consensus_50pct.png")
    print(f"\nDone. Results saved to: {out.resolve()}")
    print("Note: this prototype enforces district area constraints but does not enforce exact patch topology.")
if __name__=="__main__":
    main()
