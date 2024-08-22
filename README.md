# Rodeo: Making Refinements for Diverse Top-k Queries

This is the implementation for Rodeo as presented in the demonstration track of VLDB 2024.

## Setup Instructions

Rodeo consists of two components, a React-based frontend and a Flask-based backend which interfaces with the refinement algorithm.

Please feel free to reach out if there is any trouble setting up Rodeo.

### Frontend

```
cd frontend
npm install
```

...

### Backend

① IBM CPLEX must be installed. It can be acquired here.

② ```
  cd backend
  chmod +x run.sh
  pip install -r requirements.txt
    ```

...

## Running Rodeo

### Frontend

```
cd frontend
npm run dev
```

### Backend

```
cd backend
./run.sh
```

## Using Rodeo

...